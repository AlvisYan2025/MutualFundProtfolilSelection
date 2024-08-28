#calculate sensitivity for their models 
def make_sensitivity_graph(df):
    df = df.sort_values(by='sensitivity', ascending=False)
    plt.figure(figsize=(8, 6))
    plt.barh(df['variables'], df['sensitivity'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top Variable Importance')
    plt.show()
def get_out_of_sample_derivative(model, feature, input_timesteps, df): #
    #get outofsample derivative for one model with corresponding feature
    allfeatures = ['ages', 'flow',
        'exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no',
        'Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1',
        'F_r12_2', 'sentiment']
    diff= 0.001
    input_tensor = get_tensor_from_df(df[df['Timestep'].isin(input_timesteps)], allfeatures, False)
    output = model.predict(input_tensor)
    df_plus = df.copy()
    df_plus[feature] = df_plus[feature] + diff
    input_tensor_plus = get_tensor_from_df(df_plus[df_plus['Timestep'].isin(input_timesteps)], allfeatures, False)
    output_plus = model.predict(input_tensor_plus)
    output_diff = (output_plus - output)/diff
    print(output_diff)
    return output_diff

all_folds = chronological_sampling_scheme(469, [1,1,1])
list_features = ['ages', 'flow',
        'exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no',
        'Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1',
        'F_r12_2', 'sentiment'] 
df_features_sensitivity = pd.DataFrame(list_features, columns=['variables'])
df_features_sensitivity['sensitivity'] = None
for curr_feature in list_features:
    group_mean = []
    for i in range (len(all_their_models)): #i->group number
        df_block  = convert_to_df(all_data)
        df_block = sanitize_input(add_invalid_label_mask(df_block))
        derivative_list = []
        for block in range(3):
            curr_model = all_their_models[i][block]
            derivative = get_out_of_sample_derivative(curr_model, curr_feature, all_folds[block], df_block)
            derivative_list.append(derivative**2)
        derivative_block = torch.cat(derivative_list, dim=0)
        df_block['derivatives'] = derivative_block.flatten().numpy()
        grouped_df = df_block.groupby('Timestep')['derivatives'].mean().reset_index()
        block_mean = grouped_df['derivatives'].mean()
        group_mean.append(math.sqrt(block_mean))
    df_features_sensitivity.loc[df_features_sensitivity['variables'] == curr_feature, 'sensitivity'] = np.mean(group_mean)
#normalize across all features
sensum = df_features_sensitivity['sensitivity'].sum()
df_features_sensitivity['sensitivity_normalized'] = df_features_sensitivity['sensitivity']/sensum
make_sensitivity_graph(df_features_sensitivity)
df_features_sensitivity