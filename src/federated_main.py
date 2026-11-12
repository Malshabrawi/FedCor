#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch



from options import args_parser,get_dynamic_num_clients
from update import LocalUpdate,test_inference,train_federated_learning,federated_test_idx
from models import MLP, NaiveCNN, BNCNN, ResNet,RNN
from utils import get_dataset, average_weights, exp_details,setup_seed
from mvnt import MVN_Test
import GPR
from GPR import Kernel_GPR
from experiment_io import SystemProfile, RunRecorder






if __name__ == '__main__':
    os.environ["OUTDATED_IGNORE"]='1'
    start_time = time.time()
    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    gargs = copy.deepcopy(args)
    exp_details(args)
    if not args.iid:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_{}[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    'sp' if args.alpha is None else 'alpha',args.shards_per_client if args.alpha is None else args.alpha,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    else:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    if not os.path.exists(base_file):
        os.makedirs(base_file)
    
    if args.afl:
        file_name = base_file+'/afl'
    elif args.power_d:
        file_name = base_file+'/powerd_d[{}]'.format(args.d)
    elif args.gpr:
        file_name = base_file+'/gpr[int{}_gp{}_norm{}_disc{}]'.\
            format(args.GPR_interval,args.group_size,args.poly_norm,args.discount)
    else:
        file_name = base_file+'/random'
    
    


    

    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    gpr_device = 'cuda:'+args.gpr_gpu if args.gpr_gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(device)
    if gargs.seed is None:
        gargs.seed = [0]
    for seed in gargs.seed:
        args = copy.deepcopy(gargs)# recover the args
        print("Start with Random Seed: {}".format(seed))
        # load dataset and user groups
        train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args,seed)
        profile = SystemProfile(args.num_users, seed, args.local_ep)
        recorder = RunRecorder(args, seed, user_groups, profile)
        file_name = str(recorder.path / 'legacy')
        participation = np.zeros(args.num_users, dtype=int)
        # weights /=np.sum(weights)
        if seed is not None:
            setup_seed(seed)
        data_size = train_dataset[0][0].shape
        # BUILD MODEL
        if args.model == 'cnn':
            # Naive Convolutional neural netork
            global_model = NaiveCNN(args=args,input_shape = data_size,final_pool=False)
        
        elif args.model == 'bncnn':
            # Convolutional neural network with batch normalization
            global_model = BNCNN(args = args, input_shape = data_size)

        elif args.model == 'mlp' or args.model == 'log':
            # Multi-layer preceptron
            len_in = 1
            for x in data_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=args.mlp_layers if args.model=='mlp' else [],
                                dim_out=args.num_classes)
        elif args.model == 'resnet':
            global_model = ResNet(args.depth,args.num_classes)
        elif args.model == 'rnn':
            if args.dataset=='shake':
                global_model = RNN(256,args.num_classes)
            else:
                # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
                global_model = RNN(256,args.num_classes,300,True,128)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)

        # Build GP
        if args.gpr:
            if args.kernel=='Poly':
                gpr = Kernel_GPR(args.num_users,loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device, seed=seed,
                                    dimension = args.dimension,kernel=GPR.Poly_Kernel,order = 1,Normalize = args.poly_norm)
            elif args.kernel=='SE':
                gpr = Kernel_GPR(args.num_users,loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device,seed=seed,
                                    dimension = args.dimension,kernel=GPR.SE_Kernel)
            else:
                gpr = GPR.Matrix_GPR(args.num_users,loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device,seed=seed)
            gpr.to(gpr_device)

        if args.gpr:
            gpr.client_latencies = torch.tensor(profile.durations(
                np.arange(args.num_users), np.full(args.num_users, args.local_ep)), device=gpr.device)
            gpr.straggler_penalty = args.straggler_penalty
            gpr.preferred_duration = args.preferred_duration

        # copy weights
        global_weights = global_model.state_dict()
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded

        
        for i in range(args.num_users):
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))

        local_states = np.array(local_states)
        local_weights = np.array(local_weights)

        # Training
        train_loss, train_accuracy = [], []
        test_loss,test_accuracy = [],[]
        max_accuracy=0.0

        local_losses = []# test losses evaluated on local models(before averaging)
        chosen_clients = []# chosen clients on each epoch
        gt_global_losses = []# test losses on global models(after averaging) over all clients
        gpr_data = []# GPR Training data
        print_every = 1
        init_mu = args.mu

        
        gpr_idxs_users = None

        predict_losses = []

        sigma = []
        sigma_gt=[]


        # Test the global model before training
        list_acc, list_loss = federated_test_idx(args,global_model,
                                                list(range(args.num_users)),
                                                train_dataset,user_groups)
        gt_global_losses.append(list_loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        if args.afl:
            AFL_Valuation = np.array(list_loss)*np.sqrt(weights*len(train_dataset))

        # gpr_loss_data = None
        train_loss_history = []
        marginal_gains_history = []  # Store marginal gains for dynamic_TH
        dynamic_TH = args.dynamic_TH  # Initial value
        for epoch in tqdm(range(args.epochs)):
            if args.gpu or args.gpr_gpu:
                torch.cuda.synchronize()
            round_started = time.perf_counter()
            client_execution_seconds = 0.0
            probe_execution_seconds = 0.0
            gp_probe_client_seconds = 0.0
            print('\n | Global Training Round : {} |\n'.format(epoch+1))
            epoch_global_losses = []
            epoch_local_losses = []
            global_model.train()
            if epoch in args.schedule:
                args.lr*=args.lr_decay

            
            # Num of clinets 
            m = get_dynamic_num_clients(
            round_idx=epoch,
            total_rounds=args.epochs,
            args=args,
            train_loss_history=train_loss_history
            )
            print(f"\n dynamic_frac : {args.dynamic_frac}\t Epoch {epoch}\t Number of clients : {m} ")

            # Update dynamic_TH based on selected mode
            if args.dynamic_C and len(marginal_gains_history) > 0:
                window = args.dynamic_th_window
                if args.dynamic_th_mode == 'moving_avg':
                    recent_gains = np.array(marginal_gains_history[-window:])
                    if len(recent_gains) > 0:
                        dynamic_TH = args.dynamic_th_alpha * np.mean(recent_gains)
                elif args.dynamic_th_mode == 'percentile':
                    recent_gains = np.array(marginal_gains_history[-window:])
                    if len(recent_gains) > 0:
                        dynamic_TH = np.percentile(recent_gains, args.dynamic_th_percentile)

            if args.gpr and epoch>=args.warmup:
                # FedCor
                idxs_users, client_latencies, round_time, selected_marginal_gains = gpr.Select_Clients(m,args.epsilon_greedy,weights,args.dynamic_C,dynamic_TH,SystemAware=args.system_aware,MinClients=args.min_clients)
                print(f"Actual clients selected after dynamic threshold: {len(idxs_users)}")
                # Update Cumulative Time
                # Timing is recorded after actual adaptive epochs are known.
                print("GPR Chosen Clients:",idxs_users)
                # Store marginal gains for dynamic_TH update
                if selected_marginal_gains is not None:
                    marginal_gains_history.append(float(np.mean(selected_marginal_gains)))

            elif args.afl:
                # AFL
                delete_num = int(args.alpha1*args.num_users)
                sel_num = int((1-args.alpha3)*m)
                tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation])
                tmp_value = tmp_value[:,tmp_value[1,:].argsort()]
                prob = np.exp(args.alpha2*tmp_value[1,delete_num:])
                prob = prob/np.sum(prob)
                sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
                remain = set(np.arange(args.num_users))-set(sel1)
                sel2 = np.random.choice(list(remain),m-sel_num,replace = False)
                idxs_users = np.append(sel1,sel2)

            elif args.power_d:
                # Power-of-D-choice
                A = np.random.choice(range(args.num_users), args.d, replace=False,p=weights)
                idxs_users = A[np.argsort(np.array(gt_global_losses[-1])[A])[-m:]]

            else:
                # Random selection
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
                


            idxs_users = [int(i) for i in idxs_users]
            chosen_clients.append(idxs_users)
            client_latencies = profile.durations(idxs_users, np.full(len(idxs_users), args.local_ep))
            participation[idxs_users] += 1
            actual_epochs, local_steps, client_utilities = [], [], []

            
            
            client_started = time.perf_counter()
            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                # FIXME: arguments for the localupdate
                # try:
                if args.gpr and args.system_aware and epoch>=args.warmup:
                    local_update = LocalUpdate(args=args, 
                                            dataset=train_dataset,
                                            idxs=user_groups[idx],
                                            global_round = epoch,
                                            latency=client_latencies[idxs_users.index(idx)],
                                            preferred_T=args.preferred_duration)
                else:
                    local_update = LocalUpdate(args=args, 
                                            dataset=train_dataset,
                                            idxs=user_groups[idx],
                                            global_round = epoch)
                w,test_loss,init_test_loss = local_update.update_weights(model=local_model)
                actual_epochs.append(local_update.actual_epochs)
                local_steps.append(local_update.local_steps)
                client_utilities.append(local_update.statistical_utility)
                
                local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
                local_weights[idx]=copy.deepcopy(w)
                epoch_global_losses.append(init_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
                epoch_local_losses.append(test_loss)


            if args.gpu:
                torch.cuda.synchronize()
            client_execution_seconds = time.perf_counter() - client_started
            # update global weights
            if args.global_average:
                global_weights = average_weights(local_weights,omega=None)
            else:
                global_weights = average_weights(local_weights[idxs_users],omega=None)

            for i in range(args.num_users):
                local_weights[i] = copy.deepcopy(global_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            if args.afl:
                AFL_Valuation[idxs_users] = np.array(epoch_global_losses)*np.sqrt(weights[idxs_users]*len(train_dataset))
            local_losses.append(epoch_local_losses)

            # dynamic mu for FedProx
            loss_avg = sum(epoch_local_losses) / len(epoch_local_losses)
            if args.dynamic_mu and epoch>0:
                if loss_avg>loss_prev:
                    args.mu+=init_mu*0.1
                else:
                    args.mu=max([args.mu-init_mu*0.1,0.0])
            loss_prev = loss_avg
            train_loss.append(loss_avg)
            # if args.dynamic_frac == 3:
            #     train_loss_history.append(train_loss)
            #     if len(train_loss_history) > 20:  
            #         train_loss_history.pop(0)

            # calculate test accuracy over all users
            list_acc, list_loss = federated_test_idx(args,global_model,
                                                    list(range(args.num_users)),
                                                    train_dataset,user_groups)
            gt_global_losses.append(list_loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            
            
            # test prediction accuracy of GP model
            if args.gpr and epoch>=args.warmup:
                test_idx = np.random.choice(range(args.num_users), m, replace=False)
                test_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                            np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                            np.ones([args.num_users,1])],1)
                pred_idx = np.delete(list(range(args.num_users)),test_idx)
                
                try:
                    predict_loss,mu_p,sigma_p = gpr.Predict_Loss(test_data,test_idx,pred_idx)
                    print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
                    predict_losses.append(predict_loss)
                except (RuntimeError, ValueError) as error:
                    print("[Warning]: GP prediction diagnostic skipped: {}".format(error))
                
                

            

            # train GPR
            if args.gpr:
                if epoch>=args.gpr_begin:
                    if epoch<args.warmup:# warm-up
                        gpr.Update_Training_Data([np.arange(args.num_users),],[np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),],epoch=epoch)
                        if not args.update_mean:
                            print("Training GPR")
                            gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=150,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)
                        elif epoch == args.warmup-1:
                            print("Training GPR")
                            gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=1000,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)

                    elif epoch>=args.warmup and epoch%args.GPR_interval==0:# normal and optimization round
                        gpr.Reset_Discount()
                        print("Training with Random Selection For GPR Training:")
                        random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                        probe_started = time.perf_counter()
                        gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                                            copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
                        if args.gpu:
                            torch.cuda.synchronize()
                        probe_execution_seconds = time.perf_counter() - probe_started
                        gp_probe_client_seconds = profile.round_seconds(random_idxs_users, np.full(m, args.local_ep))
                        gpr.Update_Training_Data([np.arange(args.num_users),],[np.array(gpr_loss)-np.array(gt_global_losses[-1]),],epoch=epoch)
                        print("Training GPR")
                        gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=args.GPR_Epoch,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)

                    else:# normal and not optimization round
                        gpr.Update_Discount(idxs_users,args.discount)
                    
                
                if args.mvnt and (epoch==args.warmup-1 or (epoch%args.mvnt_interval==0 and epoch>=args.warmup)):
                    mvn_file = file_name+'/MVN/{}'.format(seed)
                    if not os.path.exists(mvn_file):
                        os.makedirs(mvn_file)
                    mvn_samples=MVN_Test(args,copy.deepcopy(global_model),
                                                train_dataset,user_groups,
                                                file_name+'/MVN/{}/{}.csv'.format(seed,epoch))
                    sigma_gt.append(np.cov(mvn_samples,rowvar=False,bias = True))
                    sigma.append(gpr.Covariance().clone().detach().numpy())

                    

            # test inference on the global test dataset
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            if args.gpu or args.gpr_gpu:
                torch.cuda.synchronize()
            measured_round_seconds = time.perf_counter() - round_started
            simulated_client_seconds = profile.round_seconds(idxs_users, actual_epochs)
            measured_server_seconds = max(0.0, measured_round_seconds-client_execution_seconds-probe_execution_seconds)
            recorder.round({'round':epoch+1,'phase':'warmup' if args.gpr and epoch<args.warmup else 'normal',
                'test_accuracy':test_acc,'test_loss':test_loss,'clients':idxs_users,
                'actual_epochs':actual_epochs,'local_steps':local_steps,'client_utilities':client_utilities,
                'client_validation_accuracy':list_acc,'client_validation_loss':list_loss,
                'participation_counts':participation.tolist(),'threshold':float(dynamic_TH) if args.dynamic_C else None,
                'simulated_client_seconds':simulated_client_seconds,'gp_probe_client_seconds':gp_probe_client_seconds,
                'measured_server_seconds':measured_server_seconds,'measured_round_seconds':measured_round_seconds})
            if args.target_accuracy is not None:
                if test_acc>=args.target_accuracy:
                    break

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
                print('Training Loss : {}'.format(np.sum(np.array(list_loss)*weights)))
                print("Test Accuracy: {:.2f}%\n".format(100*test_acc))
                # TODO: add print statement below to include wall clock time reporting per round
                print(f"Cumulative hybrid training time: {recorder.elapsed:.2f} seconds")
        
        
        print(' \n Results after {} global rounds of training:'.format(epoch+1))
        print("|---- Final Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
        print("|---- Max Test Accuracy: {:.2f}%".format(100*max(test_accuracy)))
        if args.gpr:
            print("|---- Mean GP Prediction Loss: {:.4f}".format(np.mean(predict_losses)))

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        recorder.finish(epoch+1, test_accuracy[-1])
        # save the training records:
        with open(file_name+'_{}.pkl'.format(seed), 'wb') as f:
            pickle.dump([train_loss, train_accuracy,chosen_clients,
                        weights,None if not args.gpr else gpr.state_dict(),
                        gt_global_losses,test_accuracy], f)
        
        if args.mvnt:
            with open(file_name+'/MVN/{}/Sigma.pkl'.format(seed), 'wb') as f:
                pickle.dump([sigma,sigma_gt],f)




