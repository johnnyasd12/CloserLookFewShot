import os
import pandas as pd
from matplotlib import pyplot as plt

class ExpPlotter:
    def __init__(self, csv_name, record_folder, negligible_vars, dependent_vars):
        self.csv_path = os.path.join(record_folder, csv_name)
        self.df = pd.read_csv(self.csv_path)
        self.negligible_vars = negligible_vars
        self.dependent_vars = dependent_vars
        
        self.df_drop = ExpPlotter.drop_duplicate(self.df)
        self.controllable_vars = [x for x in list(self.df_drop.columns) if x not in negligible_vars and x not in dependent_vars]
        
        
        
    def plot_exps(self, independent_var, dependent_var):
        control_vars = self.controllable_vars.copy()
        control_vars.remove(independent_var)
#         print('control_vars:', control_vars)
        
        all_settings_df = self.df_drop[control_vars].drop_duplicates()

        for _, row in all_settings_df.iterrows():
            sub_df = self.df_drop.copy()

            print('Control Variables:')
            for k,v in row.items():
                print(k, ':', v)
                if v==v: # v is not nan
                    sub_df = sub_df[sub_df[k]==v]
                else: # v is nan
                    sub_df = sub_df[sub_df[k]!=sub_df[k]]

            xs = sub_df[independent_var].values
            ys = sub_df[dependent_var].values
            print('%s:\n'%(independent_var), xs)
            print('%s:\n'%(dependent_var), ys)
            baseline = min(ys)-(max(ys)-min(ys))
            plt.bar(xs, ys-baseline, width=0.1, bottom=baseline)
            plt.show()
    
    
    def drop_duplicate(df):
        # drop columns with all the same row values
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df_drop = df.drop(cols_to_drop, axis=1)
        # drop columns with all NAN values
        df_drop = df_drop.dropna(axis=1, how='all')
        return df_drop







