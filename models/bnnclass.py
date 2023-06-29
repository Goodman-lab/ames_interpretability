#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 4 2023

BNNClassifier class with custom functions for predictions, uncertainty estimation and model interpretation.

@author: cg588
"""


import numpy as np
import scipy.stats as st

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps

class BNNClassifier(nn.Sequential): 
    
    """
    Wrapper class for Bayesian Neural Networks in PyTorch. 
    Designed for classification tasks. 
    Contains custom functions for model fitting, making predictions, uncertainty estimation and interpretation. 

    Core layers are from torchbnn (https://github.com/Harry24k/bayesian-neural-network-pytorch/)

    Example: 

    import torch
    import torchbnn as bnn
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

    model = BNNClassifier(
    bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = 8000, out_features = 10),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = 10, out_features = 10),
    nn.ReLU(), 
    bnn.BayesLinear(prior_mu = 0, prior_sigma = 0.1, in_features = 10, out_features = 2)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # convert format of X and y to Tensor objects
    X_in = torch.from_numpy(X_train).float() # need same dtype as weight matrix
    y_in = torch.from_numpy(y_train).long()

    model.set_hparams()
    model.fit(X_in, y_in)

    model.sample_posterior(X_test, n_samples = 5000)
    y_mu, y_sigma = model.predict_normal_distribution(X_test, resample = False, n_samples = 5000)

    y_pred = [int(x>0) for x in y_mu]
    print(balanced_accuracy_score(y_test, y_pred))
    print(matthews_corrcoef(y_test, y_pred))
    """


    def set_hparams(self, kl_weight = 0.1, optimizer = optim.Adam, lr = 0.01, epochs = 500): 
        # sets hyperparameters for model. do this before trying to fit. 
        self.kl_weight = kl_weight
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.epochs = epochs
    
    def fit(self, X, y): 
        # fit model parameters. 
        # X: input training data. 
        # y: input training labels. 
        # return None
        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = bnn.BKLLoss(reduction = 'mean',
                              last_layer_only=False)
        for epoch in range(self.epochs): 
            pre = self(X)
            ce = ce_loss(pre, y)
            kl = kl_loss(self)
            cost = ce + self.kl_weight*kl

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
    
    def predict_proba_once(self, X): 
        # get a single sample of the numerical output of the neural network before it is converted to predicted class. 
        # not strictly a "probability" but this is the closest interpretation to this value. 
        # X: input test data. 
        
        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        probs = self(X).data

        return (probs[:,1] - probs[:,0]).numpy()

    def predict_softmax_proba_once(self, X): 
        # convert a single sample of the numerical output of the neural network to a probability using softmax function (https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html). 
        # X: input test data. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        probs = self(X).data
        sm = nn.Softmax(dim=1)

        return sm(probs).numpy()

    def predict_once(self, X): 
        # convert a single sample of the numerical output of the neural network to a class prediction. 
        # X: input test data. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        probs = self(X).data
        _, preds = torch.max(probs, 1)
        
        return preds.numpy()

    def sample_posterior(self, X, n_samples = 1000): 
        # sample and store the posterior distribution for making predictions. 
        # this can save time if working with the same test data to avoid resampling the same posterior repeatedly. 
        # X: input test data. 
        # n_samples: number of samples to take. more samples will lead to a more credible posterior but increase computing costs. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i] = np.squeeze(self.predict_proba_once(X))
        
        self.store_x = X
        self.samples = samples
        return samples


    def predict_normal_distribution(self, X, resample = True, n_samples = 1000): 
        # get mean and standard deviation for the posterior distribution. 
        # we have normal priors over the weights in the neural network, so we expect the posterior to be a normal distribution. 
        # X: input test data. 
        # resample: whether to sample again or to take the stored samples. 
        # n_samples: number of samples to take. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        if not hasattr(self, 'store_x') or resample: 
            self.sample_posterior(X, n_samples)

        y_mu = np.array([np.mean(r) for r in self.samples])
        y_sigma = np.array([np.std(r) for r in self.samples])

        return y_mu, y_sigma

    def predict(self, X, resample = True, n_samples = 1000): 
        # predict classes for input test data. 
        # X: input test data
        # resample: whether to sample again or to take stored samples. 
        # n_samples: number of samples to take. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        if not hasattr(self, 'store_x') or resample: 
            self.sample_posterior(X, n_samples)

        y_mu = np.array([np.mean(r) for r in self.samples])

        return np.array([int(x>0) for x in y_mu])



    def predict_proba(self, X, resample = True, n_samples = 1000): 
        # predict probability of being in the positive class. 
        # obtained by considering the posterior distribution and y_proba > 0 is required to be in the positive class. 
        # X: input test data
        # resample: whether to sample again or to take stored samples. 
        # n_samples: number of samples to take. 

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        if not hasattr(self, 'store_x') or resample: 
            self.sample_posterior(X, n_samples)

        y_mu = np.array([np.mean(r) for r in self.samples])
        y_sigma = np.array([np.std(r) for r in self.samples])

        sfs = [st.norm.sf(0, y_mu[i], y_sigma[i]) for i in range(len(X))]

        return sfs


    def predict_softmax_proba(self, X, resample = True, n_samples = 1000): 
        # predict the softmax probability of being in the positive class, similar to predict_softmax_proba_once. 
        # X: input test data
        # resample: whether to sample again or to take stored samples. 
        # n_samples: number of samples to take

        if type(X) == np.ndarray: 
            X = torch.from_numpy(X).float()
        if not hasattr(self, 'store_x') or resample: 
            self.sample_posterior(X, n_samples)

        y_mu = np.array([np.mean(r) for r in self.samples])        
        
        def softmax(x): 
            return np.exp(x)/(1+np.exp(x))

        return [softmax(x) for x in y_mu]



    def config_explainer(self, explainer, sample_data): 
        # ONLY FOR SHAP
        # define which SHAP explainer to use. by default recommend DeepExplainer (although it spews a million warnings about BayesLinear)
        if type(sample_data) == np.ndarray: 
            sample_data = torch.from_numpy(sample_data).float()        
        self.explainer = explainer(self, sample_data)
        return None

    def explain_once(self, smiles, plot = False): 
        # explains model prediction using feature attributions by SHAP
        # distribute ECFP bit attributions to atoms to get atom attributions
        # smiles: input molecule to explain. need to use smiles because we need the original molecule to interpret ECFP bits. 
        # plot: whether to plot the resulting explanation. 

        if not hasattr(self, 'explainer'): 
            print('Define the explainer first with model.config_explainer before trying to explain model.')
            return None

        mol = Chem.MolFromSmiles(smiles)
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = 8000, bitInfo = bi)

        X = np.array(fp.ToList())
        shapvs = self.explainer.shap_values(torch.from_numpy(X.reshape(1,-1)).float())
        bit_contr = np.squeeze(shapvs[1] - shapvs[0])*X
        atom_weights = [0] * mol.GetNumAtoms()

        for on_bit in bit_contr.nonzero()[0]: 
            n_occurrences = len(bi[on_bit])

            for example in range(n_occurrences): 
                central_atom, radius = bi[on_bit][example]
                m_env = set((central_atom,))
                bit_path = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, central_atom)

                for atom in bit_path: 
                    m_env.add(mol.GetBondWithIdx(atom).GetBeginAtomIdx())
                    m_env.add(mol.GetBondWithIdx(atom).GetBeginAtomIdx())

                bit_weight = bit_contr[on_bit]

                for atom in m_env: 
                    atom_weights[atom] += bit_weight / n_occurrences

        if plot: 
            weights, maxweight = SimilarityMaps.GetStandardizedWeights(atom_weights)
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights)

        else: 
            return atom_weights

    def explain(self, smiles, n_samples = 1000, plot = True): 
        # explains model prediction using feature attributions by SHAP
        # sample many times to get lower variance on SHAP values
        # smiles: input molecule to explain
        # n_samples: how many samples to take
        # plot: whether to plot the resulting explanation
        
        if not hasattr(self, 'explainer'): 
            print('Define the explainer first with model.config_explainer before trying to explain model.')
            return None

        mol = Chem.MolFromSmiles(smiles)
        samples = np.zeros((mol.GetNumAtoms(), n_samples))
        for i in range(n_samples): 
            samples[:,i] = self.explain_once(smiles, plot=False)

        atom_weight_mu = np.array([np.mean(r) for r in samples])
        atom_weight_sigma = np.array([np.std(r) for r in samples])

        if plot: 
            weights, maxweight = SimilarityMaps.GetStandardizedWeights(atom_weight_mu)
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights)
            return fig

        else: 
            return atom_weight_mu, atom_weight_sigma



    def rdkit_sim_map(self, smiles, n_samples = 1000, plot = True): 
        # explains model prediction using SimilarityMaps.GetAtomicWeightsForModel from rdkit (https://www.rdkit.org/docs/source/rdkit.Chem.Draw.SimilarityMaps.html). 
        # iterates through atoms one at a time, generating fingerprints without this atom and comparing the impact on the model prediction. 
        # this process includes sampling the neural network many times, so uncertainty within the network is accounted for. 
        # smiles: input molecule to explain
        # n_samples: how many samples to take
        # plot: whether to plot the resulting explanation

        def fpstring(mol, x): 
            probefp = SimilarityMaps.GetMorganFingerprint(mol, atomId=x, radius=2, fpType='bv', nBits=8000)
            probevect = np.zeros(8000)
            DataStructs.ConvertToNumpyArray(probefp, probevect)
            X_probe = np.expand_dims(probevect, axis = 0)
            return X_probe

        def get_mean_pred(X): 
            y_mu, y_sigma = self.predict_normal_distribution(X, resample = True, n_samples = 100)
            return y_mu

        mol = Chem.MolFromSmiles(smiles)
        

        if plot: 
            fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(mol, fpstring, get_mean_pred)
            return fig
        
        else: 
            weights = SimilarityMaps.GetAtomicWeightsForModel(mol, fpstring, get_mean_pred)
            return weights