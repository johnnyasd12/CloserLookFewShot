import numpy as np
from io_utils import parse_args
from train import exp_train_val
from save_features import exp_save_features
from test import exp_test, record_csv
from param_utils import get_all_params_comb, get_all_args_ls, get_modified_args, copy_args

class ExpManager:
#     def __init__(self, base_args):
#     def __init__(self, base_args, train_fixed_params, test_fixed_params, general_possible_params, test_possible_params, record_csv=None):
    def __init__(self, base_params, train_fixed_params, test_fixed_params, general_possible_params, test_possible_params):
        '''
        Args:
            train_fixed_params (dict): e.g. {'stop_epoch':700}
            test_fixed_params (dict): e.g. {'record_csv':'program_time.csv'}
            general_possible_params (dict): dictionary of list of tunable parameters, different param would train different model. e.g. {'dropout_p':[0.25, 0.5]}
        '''
        self.base_params = base_params # general fixed params
        self.possible_params = {'general':general_possible_params, 'test':test_possible_params} # general means generalize to train/save_features/test
        self.fixed_params = {'train':train_fixed_params, 'test':test_fixed_params}
        self.results = [] # params as well as results restore in the list of dictionaries
    
    def exp_grid(self, choose_by='val_acc_mean'):
        print('exp_grid() start.')
        print(self.base_params)
        default_args = {} # the raw default args of the code
        default_args['train'] = parse_args('train', parse_str='')
        default_args['test'] = parse_args('test', parse_str='')
        possible_params = self.possible_params
        
        train_args = get_modified_args(default_args['train'], {**self.base_params, **self.fixed_params['train']})
        test_args = get_modified_args(default_args['test'], {**self.base_params, **self.fixed_params['test']})
        all_general_params = get_all_params_comb(possible_params=possible_params['general'])
        all_test_params = get_all_params_comb(possible_params=possible_params['test'])
        
        for params in all_general_params:
            modified_train_args = get_modified_args(train_args, params)
            modified_test_args = get_modified_args(test_args, params)
            # train model
            print('='*20, 'Training', '='*20)
            print(params)
            _ = exp_train_val(modified_train_args)
            
            # loop over testing settings under each general setting
            for test_params in all_test_params:
                final_test_args = get_modified_args(modified_test_args, test_params)
                
                splits = ['val', 'novel'] # temporary no 'train'
                write_record = {**params, **test_params}
                for split in splits:
                    split_final_test_args = copy_args(final_test_args)
                    split_final_test_args.split = split
                    print('='*20, 'Saving Features', '='*20)
                    print(params)
                    print(test_params)
                    print(split)
                    exp_save_features(copy_args(split_final_test_args))
                    print('='*20, 'Testing', '='*20)
                    record = exp_test(copy_args(split_final_test_args), iter_num=600)
                    write_record['epoch'] = record['epoch']
                    write_record[split+'_acc_mean'] = record['acc_mean']
                
                self.results.append(write_record)
                record_csv(final_test_args, write_record, csv_path='./record/'+final_test_args.csv_name)
                
        # TODO: can also loop dataset
        for choose_by in ['val_acc_mean', 'novel_acc_mean']:
            sorted_result = sorted(self.results, key = lambda i: i[choose_by], reverse=True)
            best_result = sorted_result[0]
            print(best_result)
            print('The best test acc is', best_result['novel_acc_mean'],'%% on grid search chosen by:',choose_by)
        
    def exp_grid_search(dataset, split): # choose the best according to dataset & split
        pass
    
    def exp_random_search():
        pass



def main(args):
    pass


if __name__=='__main__':
    args = None
    main(args)






