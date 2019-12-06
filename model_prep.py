class ModelPrepTransFeatures(object):
    '''
    Class methods to transform features.  Constructs the dataframe to transform.
    '''

    def __init__(self, df):
        self.df_c = df.copy()

    def nulls_numeric(self, thresh=.05) -> 'index':
        '''
        Method to plug nulls for numeric dtypes
        Args: 
            Thresh: less than threshold to use for selecting columns to process
        Returns: Numeric columns with less than the threshold
        '''
        nulls_numeric = self.df_c.select_dtypes(['number']).isnull().sum().apply(lambda x: x / len(self.df_c))
        nulls_to_plug = nulls_numeric[(nulls_numeric < thresh) & (nulls_numeric > 0)].index
        return nulls_to_plug

    def mean_plug(self) -> 'df':  # mean plug nulls with less than 5%
        '''
        Method to plug in nulls with means for columns defined in nulls_numeric method
        Returns: dataframe with mean plugged columns
        '''
        nulls_to_plug = self.nulls_numeric()
        self.df_c[nulls_to_plug] = self.df_c[nulls_to_plug].fillna(self.df_c.mean())
        return self.df_c

    def mode_plug(self) -> 'df':  # mode plug nulls with less than 5%
        '''
        Method to plug in nulls with mode for columns defined in nulls_numeric method
        Returns: dataframe with mode plugged columns
        '''
        nulls_to_plug = self.nulls_numeric()
        self.df_c[nulls_to_plug] = self.df_c[nulls_to_plug].fillna(self.df_c.mode().loc[0])
        return self.df_c

    def create_dummies(self) -> 'df and index':  # create dummy for remaining non-numeric columns
        '''
        Method to replace non-numeric columns with numeric dummies
        Returns:
            dataframe with dummy columns
            columns replaced with dummies
        '''
        import pandas as pd
        object_cols = self.df_c.dtypes[self.df_c.dtypes == 'object'].index
        if len(object_cols) > 0:
            dummies = pd.get_dummies(self.df_c[object_cols])
            self.df_c = self.df_c.drop(object_cols, axis=1)
            self.df_c = pd.concat([self.df_c, dummies], axis=1)
        return self.df_c, object_cols


class ModelPrepSelectFeatures(object):
    '''
    Class to select features for modeling.  Construct the dataframe to process.
    '''

    def __init__(self, df):
        self.df_c = df.copy()

    def drop_low_var_vars(self) -> 'df and index':  # drop low variance columns
        '''
        Method to drop low variance variables defined as less than .01 variance.
        Returns:
            dataframe with dropped columns
            index of columns dropped
        '''
        numeric_cols = self.df_c.select_dtypes(['number']).columns
        df_c_num = self.df_c[numeric_cols]
        df_c_normalized = (df_c_num - df_c_num.min()) / (df_c_num.max() - df_c_num.min())
        df_c_var = df_c_normalized.var().sort_values()
        low_var_to_drop = df_c_var[df_c_var < .01].index
        self.df_c = self.df_c.drop(low_var_to_drop, axis=1)
        return self.df_c, low_var_to_drop

    def drop_nulls(self, thresh=.25) -> 'df and index':  # cols with 25% or more of nulls
        '''
        Method to drop nulls.
        Args:
            thresh: columns with greater than this nulls to drop
        Returns:
            dataframe with dropped columns
            index of columns dropped
        '''
        nulls = self.df_c.isnull().sum().apply(lambda x: x / len(self.df_c))
        nulls_dropped = nulls[nulls > thresh].index
        self.df_c = self.df_c.drop(nulls_dropped, axis=1)
        return self.df_c, nulls_dropped

    def collin(self, thresh=.8) -> 'dict':
        '''
        Method to identify multicollinear variables with high correlation to each other.
        Args:
            thresh: threshold to consider for high correlation
        Returns:
            dictionary of collinear variables
        '''
        from copy import deepcopy
        corr = self.df_c.corr()
        corr_col = corr.columns
        corr_dict_0 = {}
        for i in corr_col:
            high_corr_val = list(corr[i][(corr[i] > thresh) | (corr[i] < -thresh)])
            high_corr_val = [f'{element:.4f}' for element in high_corr_val]
            high_corr_var = list(corr[i][(corr[i] > thresh) | (corr[i] < -thresh)].index)
            if i in high_corr_var:
                ind = high_corr_var.index(i)
                high_corr_var.remove(i)
                del high_corr_val[ind]
            corr_dict_0[i] = list(zip(high_corr_var, high_corr_val))
        corr_dict = deepcopy(corr_dict_0)
        for k, v in corr_dict_0.items():
            if len(v) == 0:
                corr_dict.pop(k)
        del corr_dict_0
        return corr_dict

    def correlation_heatmap(self) -> 'nothing returned, heatmape displayed':
        '''
        Method to display a heatmap of multicollinearity
        '''
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        corr = self.df_c.corr()

        sns.set(style="white")
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    def drop_low_corr_vars(self, target,
                           thresh=.3) -> 'df':  # drop columns that have less than 30% correlation to target
        '''
        Method to drop variables with very low correlation to target.
        Args:
            target: target variable
            thresh: threshold for low correlation
        Returns:
            dataframe with dropped columns
        '''
        corr = self.df_c.corr()[target].abs().sort_values(ascending=False)
        corr_drop = corr[corr <= thresh].index
        self.df_c = self.df_c.drop(corr_drop, axis=1)
        return self.df_c

    def drop_skewed_vars(self, thresh=.9) -> 'df':  # drop columns where one variable is greater than 90%
        '''
        Method to drop variables that are highly skewed to one variable.
        Args:
            thresh: threshold greater to consider as highly skewed
        Returns: 
            dataframe with dropped columns
        '''
        object_cols_before_drop = self.df_c.dtypes[self.df_c.dtypes == 'object'].index
        skewed_vars = {}
        for i in object_cols_before_drop:
            top_freq = self.df_c[i].value_counts(normalize=True)[0]
            if top_freq > .9:
                skewed_vars[i] = top_freq
        skewed_vars = list(skewed_vars.keys())
        self.df_c = self.df_c.drop(skewed_vars, axis=1)
        return self.df_c

    def drop_outliers(self, target, whis_thresh=1.5) -> 'df':  # drop outliers
        '''
        Method to drop outliers
        Args:
            target: target variables
            whis_thresh: whisker level to determine outliers
        Returns:
            datafrom with dropped columns
        '''
        whis = whis_thresh
        bottom25 = self.df_c[target].describe()['25%']
        top50 = self.df_c[target].describe()['50%']
        top25 = self.df_c[target].describe()['75%']
        top_outlier = top25 + (top25 - bottom25) * whis
        bottom_outlier = bottom25 - (top25 - bottom25) * 1.5
        outlier_index = self.df_c[self.df_c[target] > top_outlier].index
        self.df_c = self.df_c.drop(outlier_index)
        return self.df_c

    # def train_and_test(df,target,k = 0):
    #     import pandas as pd

    #     def score(train,test):
    #         lr = LinearRegression()
    #         lr.fit(train.drop(target,axis=1),train[target])
    #         var_imp = list(zip(lr.coef_, train.drop(target,axis=1)))
    #         prediction = lr.predict(test.drop(target,axis=1))
    #         rmse = mean_squared_error(test[target],prediction)**(1/2)
    #         return rmse,var_imp    

    #     np.random.seed(1)
    #     df_numeric = df
    #     if k == 0:
    #         df_recs = int(len(df_numeric)/2)
    #         train = df_numeric.iloc[:df_recs]
    #         test = df_numeric.iloc[df_recs:]
    #         result = score(train,test)
    #         return result
    #     elif k == 1:
    #         df_numeric = df_numeric.reindex(np.random.permutation(df_numeric.index))
    #         df_recs = int(len(df_numeric)/2)
    #         train = df_numeric.iloc[:df_recs]
    #         test = df_numeric.iloc[df_recs:]
    #         result = score(train,test)
    #         return result
    #     else:
    #         df_numeric = df_numeric.reindex(np.random.permutation(df_numeric.index))
    #         fold_num = int(len(df_numeric)/k)
    #         for i in range(1,k+1):
    #             if i  == 1:
    #                 df_numeric.loc[df_numeric.index[:fold_num],'fold'] = i
    #             else:
    #                 df_numeric.loc[df_numeric.index[fold_num*(i-1):fold_num*i],'fold'] = i
    #         df_numeric['fold'] = df_numeric['fold'].fillna(k)
    #         df_numeric['fold'] = df_numeric['fold'].fillna(k)
    #         rmses=[]
    #         for i in range(1,k+1):
    #             target = 'SalePrice'
    #             train = df_numeric[df_numeric['fold']!=i]
    #             test = df_numeric[df_numeric['fold']==i]
    #             train = train.drop('fold',axis = 1)
    #             test = test.drop('fold',axis = 1)
    #             result = score(train,test)
    #             rmses.append(result)
    #         return rmses

    # def train_and_test_kf(df,folds):
    #     df_numeric = df
    #     model = LinearRegression()
    #     target = 'SalePrice'
    #     kf = KFold(n_splits=folds,shuffle=True,random_state=1)
    #     mse = cross_val_score(model, df_numeric.drop(target,axis=1), df_numeric[target],                                scoring="neg_mean_squared_error", cv=kf)
    #     rmse = np.sqrt(np.abs(mse))
    #     avg_rmse = np.mean(rmse)
    #     std_rmse = np.std(rmse)
    #     error = {'avg':avg_rmse,'std':std_rmse}

    #     return error

    # # In[ ]:

# scoring

# coef = run_model[1]
# intercept = lr.intercept_
# print(coef)
# print(intercept)

# scores = []
# for i,r in test.iterrows():
#     row_score = 0
#     for f in coef:
#         row_score += r[f[1]]*f[0]
#     row_score += intercept

#     scores.append(row_score)

# predicted = pd.Series(predidction)
# features['pred_price'] = predicted
# features[['SalePrice','pred_price']]
