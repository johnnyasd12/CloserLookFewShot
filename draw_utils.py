import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class ExpPlotter:
    def __init__(self, csv_name, record_folder, base_params, negligible_vars, dependent_vars):
        '''
        Args:
            csv_name        (str)  : record file name (exclude file path)
            record_folder   (str)  : record file folder
            base_params     (dict) : only consider this setting
            negligible_vars (list) : ignore these variables
            dependent_vars  (list) : result variables
        '''
        
        self.csv_path = os.path.join(record_folder, csv_name)
        self.df = pd.read_csv(self.csv_path)
        
        self.base_params = base_params
        self.negligible_vars = negligible_vars
        self.dependent_vars = dependent_vars
        
        self.df_drop = ExpPlotter.get_matched_df(base_params, self.df)
        self.df_drop = ExpPlotter.drop_duplicate(self.df_drop)
        
        self.controllable_vars = [x for x in list(self.df_drop.columns) if x not in negligible_vars and x not in dependent_vars] # contains independent variable & controlled variables
        
    def get_matched_df(params, df):
        for k,v in params.items():
            if v==None or v!=v: # nan
                df = df[df[k]!=df[k]]
            else:
                df = df[df[k]==v]
        return df
    
    def show_best_results(self, choose_by, top_k):
        '''
        Args:
            choose_by (str) : val_acc_mean, novel_acc_mean
            top_k (int)     : int or None
        '''
        sorted_df = self.df_drop.sort_values(by=choose_by, ascending=False)
#         compare_cols = list(self.possible_params['general'].keys())+list(self.possible_params['test'].keys())
#         compare_cols = compare_cols + ['val_acc_mean', 'novel_acc_mean']
        print()
        print('Best Test Acc: %s, selected by %s'%(sorted_df['novel_acc_mean'].iloc[0], choose_by))
        print()
        print('='*20,'Top %s results sorted by: %s of all %s experiments'%(top_k, choose_by, len(sorted_df)), '='*20)
        if top_k is None:
            print(sorted_df)
        else:
            print(sorted_df.head(top_k))
    
    def plot_exps(self, independent_var, dependent_var, specific=True):
        
        def process_nan_xs(xs, ys, mode):
            is_nan = np.isnan(xs)
            if any(is_nan):
                nan_id = np.argwhere(is_nan)[0][0] # dunno why 2 dims, whatever...
                print('nan process mode:', mode)
                print('nan_id:', nan_id)
                if mode == 'delete':
                    xs = np.delete(xs, nan_id)
                    ys = np.delete(ys, nan_id)
                elif mode == 'replace':
                    xs[nan_id] = 0
                else:
                    raise ValueError('Invalid mode: %s'%(mode))
                    
            return xs, ys
        
        def get_y_baseline(ys):
            baseline = min(ys)-2 #min(ys)-(max(ys)-min(ys))
            return baseline
        
        def get_barwidth(xs):
#             bar_width = (np.nanmax(xs)-np.nanmin(xs))/(len(xs)+3)
            bar_width = (np.nanmax(xs)-np.nanmin(xs))/len(xs) * 0.5
            return bar_width
        
        print(self.base_params)
        
        control_vars = self.controllable_vars.copy()
        control_vars.remove(independent_var)
        nan_x_process_mode = 'replace'
        
        if not specific:
            possible_values = self.df_drop[independent_var].drop_duplicates().values
            mean_dependent_values = []
            df = self.df_drop.copy()
            for independent_value in possible_values:
                if independent_value==independent_value: # not nan
                    sub_df = df[df[independent_var]==independent_value]
                else: # nan
                    sub_df = df[df[independent_var]!=df[independent_var]]
                mean_value = sub_df[dependent_var].mean()
                mean_dependent_values.append(mean_value)
            
            xs = np.asarray(possible_values)
            ys = np.asarray(mean_dependent_values)
            xs, ys = process_nan_xs(xs, ys, mode=nan_x_process_mode)
            
            print('%s:\n'%(independent_var), xs)
            print('%s:\n'%(dependent_var), ys)
            
            baseline = get_y_baseline(ys)
            bar_width = get_barwidth(xs)
            print('baseline:', baseline)
            print('bar_width:', bar_width)
            plt.bar(xs, ys-baseline, width=bar_width, bottom=baseline)
            plt.show()
            
        else:
            all_settings_df = self.df_drop[control_vars].drop_duplicates()

            for _, setting_row in all_settings_df.iterrows():
                sub_df = self.df_drop.copy()

                print('='*20, 'Control Variables:', '='*20)
                for k,v in setting_row.items():
                    print(k, ':', v)
                    if v==v: # v is not nan
                        sub_df = sub_df[sub_df[k]==v]
                    else: # v is nan
                        sub_df = sub_df[sub_df[k]!=sub_df[k]]

                xs = sub_df[independent_var].values
                ys = sub_df[dependent_var].values
                xs, ys = process_nan_xs(xs, ys, mode=nan_x_process_mode)
                
                print('%s:\n'%(independent_var), xs)
                print('%s:\n'%(dependent_var), ys)
                
                baseline = get_y_baseline(ys)
                bar_width = get_barwidth(xs)
                plt.bar(xs, ys-baseline, width=bar_width, bottom=baseline)
                plt.show()
    
    
    def drop_duplicate(df):
        # drop columns with all the same row values
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df_drop = df.drop(cols_to_drop, axis=1)
        # drop columns with all NAN values
        df_drop = df_drop.dropna(axis=1, how='all')
        return df_drop







