# Externals
import copy
import tqdm
import numpy as np
import pandas as pd
# import statsmodels.api as sm 
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Locals
import config



def fit_reg(degree, df_train, df_val, train_df, val_df, features_list, interactions=False):
    """
    Fit linear, quadratic, or cubic regression.
    
    Parameters:
    degree (int) : polynomial degree (2, 3,... etc.)
    df_train : trimmed and cleaned version of train_df (subset to key features and dropped NANs)
    df_val : trimmed and cleaned version of val_df (subset to key features and dropped NANs)
    train_df : cleaned train set (after data pre-processing)
    val_df : cleaned val set (after data pre-processing)
    """
    # get global variable
    features = config.features
    
    if degree == 1 and interactions == False:
        ffeatures = ' + '.join(str(v) for v in features)
    elif degree == 2 and interactions == False:
        ffeatures = ' + '.join(str(v) for v in features) + ' + ' + ' + '.join(str(v) for v in [f'I({i}**2)' for i in features])
    elif degree == 3 and interactions == False:
        ffeatures = ' + '.join(str(v) for v in features) + ' + ' + ' + '.join(str(v) for v in [f'I({i}**2)' for i in features]) + ' + ' + ' + '.join(str(v) for v in [f'I({i}**3)' for i in features])
        
    # Fit model
    model = sm.ols(formula = f'{config.output} ~ {ffeatures}', data = df_train).fit()
    
    # Predict on train set and append to main df
    df_train_sub = df_train[features_list + [config.output]]
    X_train = df_train_sub[features_list]
    y_train = df_train_sub[config.output]
    pm_dswe_train = model.predict(X_train)    # predict
    if degree == 1 and interactions == False:
        df_train_sub[f'pred_{config.output}_LM'] = pm_dswe_train    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        train_df = pd.concat([train_df, df_train_sub[f'pred_{config.output}_LM']], axis=1)
    elif degree == 2 and interactions == False:
        df_train_sub[f'pred_{config.output}_P2M'] = pm_dswe_train    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        train_df = pd.concat([train_df, df_train_sub[f'pred_{config.output}_P2M']], axis=1)
    elif degree == 3 and interactions == False:
        df_train_sub[f'pred_{config.output}_P3M'] = pm_dswe_train    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        train_df = pd.concat([train_df, df_train_sub[f'pred_{config.output}_P3M']], axis=1)

    # Predict on val set and append to main df
    df_val_sub = df_val[features_list + [config.output]]
    X_val = df_val_sub[features_list]
    y_val = df_val_sub[config.output]
    pm_dswe_val = model.predict(X_val)    # predict
    if degree == 1 and interactions == False:
        df_val_sub[f'pred_{config.output}_LM'] = pm_dswe_val    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        val_df = pd.concat([val_df, df_val_sub[f'pred_{config.output}_LM']], axis=1)
    elif degree == 2 and interactions == False:
        df_val_sub[f'pred_{config.output}_P2M'] = pm_dswe_val    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        val_df = pd.concat([val_df, df_val_sub[f'pred_{config.output}_P2M']], axis=1)
    elif degree == 3 and interactions == False:
        df_val_sub[f'pred_{config.output}_P3M'] = pm_dswe_val    # combine with main df
        # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
        val_df = pd.concat([val_df, df_val_sub[f'pred_{config.output}_P3M']], axis=1)
    
    
    return model, train_df, val_df



def train_rf(df_train, df_val, train_df, val_df, features_list, nestimators=100, randomstate=42):
    """
    Train random forest model.
    
    Parameters:
    df_train : trimmed and cleaned version of train_df (subset to key features and dropped NANs)
    df_val : trimmed and cleaned version of val_df (subset to key features and dropped NANs)
    train_df : cleaned train set (after data pre-processing)
    val_df : cleaned val set (after data pre-processing)
    """
    
    X = df_train[features_list]   # X = df[['temp']]
    y = df_train[config.output]    
    
    # Instantiate model with 100 decision trees
    model = RandomForestRegressor(n_estimators = nestimators, random_state = randomstate)
    # Train the model on training data
    model.fit(X, y)
    
    # Predict on train set and append to main df
    df_train_sub = df_train[features_list + [config.output]]
    X_train = df_train_sub[features_list]
    y_train = df_train_sub[config.output]
    rf_dswe_train = model.predict(X_train)    # predict
    df_train_sub[f'pred_{config.output}_RF'] = rf_dswe_train    # combine with main df
    # combine lm predictions to full train set (NANs will then be kept for dates and sites with NANs in the rows of interest)
    train_df = pd.concat([train_df, df_train_sub[f'pred_{config.output}_RF']], axis=1)
    
    # Predict on val set and append to main df
    df_val_sub = df_val[features_list + [config.output]]
    X_val = df_val_sub[features_list]
    y_val = df_val_sub[config.output]
    rf_dswe_val = model.predict(X_val)       # predict
    df_val_sub[f'pred_{config.output}_RF'] = rf_dswe_val       # put lm predictions into df
    # combine lm predictions to full val set (NANs will then be kept for dates and sites with NANs in the rows of interest)
    val_df = pd.concat([val_df, df_val_sub[f'pred_{config.output}_RF']], axis=1)
    
    return model, train_df, val_df



def neural_net(X_train, y_train, X_val, y_val, featuresList, nnodes, lrate=0.00001, bsize=256, nepochs=1500):
    '''
    Define and train neural network.
    '''
    
    n_epochs = nepochs
    lr = lrate
    bs = bsize
    node = nnodes   # node = 256, 512, 1024

    # define model
    model = nn.Sequential(
        nn.Linear(len(featuresList), node),
        nn.LeakyReLU(),
        nn.Linear(node, 1),
    )

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # batches
    batch_start = torch.arange(0, len(X_train), bs)

    # Hold the best model
    best_mse = np.inf   # initialize with infinity
    best_weights = None
    train_losses = []
    history_val = []
    history_train = []

    # training
    for epoch in range(n_epochs):
        # print(epoch)
        epoch_losses = []
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+bs]
                y_batch = y_train[start:start+bs].unsqueeze(1)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_losses.append(loss.detach().numpy().item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress for reference
                bar.set_postfix(mse=float(loss))
        train_losses.append(np.mean(epoch_losses))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred_train = model(X_train)
        mse_train = loss_fn(y_pred_train, y_train.unsqueeze(1))
        history_train.append(float(mse_train))

        y_pred = model(X_val)
        mse = loss_fn(y_pred, y_val.unsqueeze(1))
        mse = float(mse)
        history_val.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    final_weights = copy.deepcopy(model.state_dict())
    
    return model, best_weights, final_weights, mse_train, mse, history_train, history_val



def train_nn(df_train, df_val, train_df, val_df, features_list, l1_nnodes, lrate=0.00001, bsize=256, nepochs=1500):
    '''
    Data manipulation pre- and post- training of neural network.
    
    Parameters:
    df_train : trimmed and cleaned version of train_df (subset to key features and dropped NANs)
    df_val : trimmed and cleaned version of val_df (subset to key features and dropped NANs)
    train_df : cleaned train set (after data pre-processing)
    val_df : cleaned val set (after data pre-processing)    
    '''
    
    X = df_train[features_list]
    y = df_train[config.output]
    
    
    df_val_sub = df_val[features_list + [config.output]]
    
    X_val = df_val_sub[features_list]
    y_val = df_val_sub[config.output]

    
    # train-val split for model evaluation
    X_train_raw, X_val_raw, y_train, y_val = X, X_val, y, y_val

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # Convert to torch.Tensor
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_train = torch.from_numpy(np.array(y_train)).float()
    y_val = torch.from_numpy(np.array(y_val)).float()
    
    model, best_weights, final_weights, _, _, train_mses, val_mses = neural_net(X_train, y_train, X_val, y_val, features_list, nnodes=l1_nnodes, nepochs=nepochs)
    
    '''
    # plot losses
    plt.plot(train_mses, label='train loss')
    plt.plot(val_mses, label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'MSELoss Over Epochs \n model LR: {lrate}, Nodes: {l1_nnodes}, BS: {bsize}')
    plt.show()
    # plt.savefig(f'{save_dir}lr_{lrate}_nodes_{l1_nnodes}_bs_{bsize}_LOSS.png', dpi=300)
    # plt.clf()
    # plt.close('all')
    '''
    
    model.load_state_dict(best_weights)
    # model.load_state_dict(final_weights) 
    
    model.eval()
    
    # train set
    df_train_sub = df_train[features_list + [config.output]]
    # put lm predictions into df
    df_train_sub[f'pred_{config.output}_NN'] = model(X_train).detach().numpy()
    # combine lm predictions to full val set (NANs will then be kept for dates and sites with NANs in the rows of interest)
    train_df = pd.concat([train_df, df_train_sub[f'pred_{config.output}_NN']], axis=1)
    
    
    # val set
    # put lm predictions into df
    df_val_sub[f'pred_{config.output}_NN'] = model(X_val).detach().numpy()
    # combine lm predictions to full val set (NANs will then be kept for dates and sites with NANs in the rows of interest)
    val_df = pd.concat([val_df, df_val_sub[f'pred_{config.output}_NN']], axis=1)
    
    
    return model, scaler, train_df, val_df



def recreate_scaler(df_train, df_val, train_df, val_df, features_list, l1_nnodes, lrate=0.00001, bsize=256, nepochs=1500):
    '''
    When reading in saved neural network from file, need to recreate the standard scalar.
    '''
    
    X = df_train[features_list]
    y = df_train[config.output]
    
    
    df_val_sub = df_val[features_list + [config.output]]
    
    X_val = df_val_sub[features_list]
    y_val = df_val_sub[config.output]

    
    # train-val split for model evaluation
    X_train_raw, X_val_raw, y_train, y_val = X, X_val, y, y_val

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    return scaler, train_df
    