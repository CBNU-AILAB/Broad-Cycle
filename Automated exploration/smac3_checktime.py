import datetime
import os
import random
import torch
import torch.nn as nn
from torchsummary import summary
import pdb
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plot
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import pylab as plt
import pandas as pd
from pyts.datasets import fetch_ucr_dataset, ucr_dataset_list
import time
import warnings
import numpy as np
from ConfigSpace import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    NormalFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning

from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import PriorAcquisitionFunction

from imgcoding.coding import Image_coding
from models.broadcycle import CycleNet, CycleMLP
from utils.utils import (
    count_parameters,
    min_max,
    z_score,
    stacking
)
from utils.customdata import CustomDataset
__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

"""
cd beom
source .virtualenvs/please/bin/activate

"""

class Babroad:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()
        RP = CategoricalHyperparameter(
            "RP",
            ["rp", "none"],
            default_value = "rp",
        )
        RERP = CategoricalHyperparameter(
            "RERP",
            ["rerp", "none"],
            default_value = "none",
        )
        TRRP = CategoricalHyperparameter(
            "TRRP",
            ["trrp", "none"],
            default_value = "none",
        )
        GAS = CategoricalHyperparameter(
            "GAS",
            ["gas", "none"],
            default_value = "gas",
        )
        REGAS = CategoricalHyperparameter(
            "REGAS",
            ["regas", "none"],
            default_value = "none",
        )
        TRGAS = CategoricalHyperparameter(
            "TRGAS",
            ["trgas", "none"],
            default_value = "none",
        )
        GAD = CategoricalHyperparameter(
            "GAD",
            ["gad", "none"],
            default_value = "gad",
        )
        REGAD = CategoricalHyperparameter(
            "REGAD",
            ["regad", "none"],
            default_value = "none",
        )
        TRGAD = CategoricalHyperparameter(
            "TRGAD",
            ["trgad", "none"],
            default_value = "none",
        )
        MK = CategoricalHyperparameter(
            "MK",
            ["mk", "none"],
            default_value = "none",
        )
        REMK = CategoricalHyperparameter(
            "REMK",
            ["remk", "none"],
            default_value = "none",
        )
        TRMK = CategoricalHyperparameter(
            "TRMK",
            ["trmk", "none"],
            default_value = "none",
        )
        CTW = CategoricalHyperparameter(
            "CTW",
            ["ctwav", "none"],
            default_value = "none",
        )
        # We do not have an educated belief on the number of layers beforehand
        # As such, the prior on the HP is uniform
        n_stage = UniformIntegerHyperparameter(
            "n_stage",
            lower=2,
            upper=4,
            default_value = 4,
        )

        # We believe the optimal network is likely going to be relatively wide,
        # And place a Beta Prior skewed towards wider networks in log space
        n_cells = UniformIntegerHyperparameter(
            "n_cells",
            lower=1,
            upper=3,
            default_value = 3,
        )
        n_dims = UniformIntegerHyperparameter(
            "n_dims",
            lower=12,
            upper=96,
            default_value = 24,
        )

        # We believe that ReLU is likely going to be the optimal activation function about
        # 60% of the time, and thus place weight on that accordingly

        # Add all hyperparameters at once:
        cs.add_hyperparameters([n_stage, n_cells, n_dims, RP, RERP, TRRP, GAS, REGAS, TRGAS, GAD, REGAD, TRGAD, MK, REMK, TRMK, CTW])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            infeature_list = []   # 선택된 입력 특징 저장되는 리스트
            if config["RP"] != "none":
                infeature_list.append(config["RP"])
            if config["RERP"] != "none":
                infeature_list.append(config["RERP"])
            if config["TRRP"] != "none":
                infeature_list.append(config["TRRP"])
            if config["GAS"] != "none":
                infeature_list.append(config["GAS"])
            if config["REGAS"] != "none":
                infeature_list.append(config["REGAS"])
            if config["TRGAS"] != "none":
                infeature_list.append(config["TRGAS"])
            if config["GAD"] != "none":
                infeature_list.append(config["GAD"])
            if config["REGAD"] != "none":
                infeature_list.append(config["REGAD"])
            if config["TRGAD"] != "none":
                infeature_list.append(config["TRGAD"])
            if config["MK"] != "none":
                infeature_list.append(config["MK"])
            if config["REMK"] != "none":
                infeature_list.append(config["REMK"])
            if config["TRMK"] != "none":
                infeature_list.append(config["TRMK"])
            if config["CTW"] != "none":
                infeature_list.append(config["CTW"])
            # print(infeature_list)
            channel_num = len(infeature_list)
            train_stack = stacking(input_size, train_size, channel_num, infeature_list)
            test_stack = stacking(input_size, train_size, channel_num, infeature_list)
            train_all = train_stack(train_dict)
            test_all = test_stack(test_dict)

            train_data = CustomDataset(train_all, ucr_target_train)
            trainloader = DataLoader(train_data, batch_size = 5, shuffle = True)
            test_data = CustomDataset(test_all, ucr_target_test)
            testloader = DataLoader(test_data, batch_size =5, shuffle = False )

            stage = config["n_stage"]
            cell = config["n_cells"]
            dims = config["n_dims"]
            cells = [] # net 구조 위한 리스트
            embed_dims = [] # net 구조 위한 리스트
            if dims%2 != 0:
                dims = dims +1
            for s in range(stage):
                if s == 0 :
                    cells = cell
                    embed_dims = dims
                else :
                    add_cell = cell
                    dims = 2*dims
                    cells = np.append(cells, add_cell)
                    embed_dims = np.append(embed_dims, dims)

            cells = cells.tolist()
            embed_dims = embed_dims.tolist()
            # print("embed_dims : ", embed_dims)
            # print("cells : ", cells)
            transitions = [True, True, True, True]
            mlp_ratios = [4, 4,4,4]
            offset_size = [16, 8, 4, 2]
            model = CycleNet(cells, channel_num, embed_dims=embed_dims, patch_size=7, transitions=transitions, num_classes=class_num,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, offset_size = offset_size).to(device)
            parameter_num = count_parameters(model)
            # print("parameter_num : ", parameter_num )
            macs, params = get_model_complexity_info(model, (channel_num, 64,64), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
            # print("flops : ", macs*2)
            string_macs, params = get_model_complexity_info(model, (channel_num, 64,64), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
            # print("macs : ", string_macs)
            # if (parameter_num < 2300000) and (parameter_num>1700000):
            #     print("skip")
            #     return 100
            real_epochs = 0
            if (parameter_num > 2340000):
                print("skip")
                return 100
            if (parameter_num < 500000 ):
                real_epochs = 50
            elif (parameter_num >500000) and (parameter_num<1200000):
                real_epochs = 100
            elif (parameter_num>1200000):
                real_epochs = 150
            print("infeature_list : ",infeature_list)
            print("cells : ", cells)
            print("embed_dims : ", embed_dims)
            print("parameter_num : ", parameter_num )
            print("macs : ", string_macs)
            print("flops : ", macs*2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = 0.0001)
            loss_list = []
            iter = []
            train_accuracy = []
            validation_loss = []
            validation_acc = []
            model.train()
            epochs = 1
            print("epoch:", epochs)
            print("real_epochs:",real_epochs)
            train_start = time.time()
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0.0
                total = 0.0
                for data in trainloader:
                    input,target = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    output = model(input)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    _, preds = torch.max(output.data, 1)
                    correct += preds.eq(target).sum().item()
                    running_loss += loss.item()
            training_loss = running_loss/len(trainloader)
            train_acc = 100*correct/len(trainloader.dataset)
            train_end = time.time()
            train_total = (train_end-train_start)*real_epochs
            global trtime_list
            trtime_list.append(train_total)
            print("train_time : ", train_total)
            train_time = datetime.timedelta(seconds=train_total)
            print("train_time : ", train_time)
            #print("=============================================================================================================================================================")
            print(f"TRAIN: EPOCH {epoch + 1:04d} / {epochs:04d} | Epoch LOSS {training_loss:.4f} | Epoch ACC {train_acc:.4f}")
            cost = running_loss/6
            loss_list.append(training_loss)
            global candi
            candi = candi + 1
            #torch.save(model.state_dict(), PATH +str(candi) +'trainPhalangesOutlinesCorrect(smac3)1.pt')
            train_accuracy.append(train_acc)
            running_loss = 0.0
            print('finish')
            correct = 0
            total = 0
            accuracy = []
            iter_acc = []
            test_loss = []
            b=0
            global img_test_acc
            model.eval()
            test_start = time.time()
            with torch.no_grad():
                for data in testloader :
                    b+=1
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    iter_acc.append(b)
                    cost = 100*correct/total
                    accuracy.append(cost)

                test_loss.append(loss)
                error_rate = 1-(correct/total)
                test_acc = 100. * correct / len(testloader.dataset)
            test_end = time.time()
            test_time = (test_end-test_start)
            global tetime_list, total_time_list, time_count
            tetime_list.append(test_time)
            print("test_time : ", test_time)
            test_time_result = datetime.timedelta(seconds=test_time)
            total_time = test_time+train_total
            print("test_time : ", test_time_result)
            print("total_time : ", total_time)
            total_time_list.append(total_time)
            time_count = time_count+1
            if img_test_acc<test_acc:
                img_test_acc = test_acc
                #torch.save(model.state_dict(), PATH + 'PhalangesOutlinesCorrect(smac3)1.pt')
                print("save best test")
            #print('accuracy of testimages: %d %%' %(100*correct/total))
            print('accuracy of testimages: %d %%' %(test_acc))
            print('best_acc : ', img_test_acc)
            print('error rate : ', format(error_rate, ".3f"))
            print("=============================================================================================================================================================")
        return 100-test_acc

stage_list = []
cell_list = []
cell_dim_list = []
time_list = []
acc_list = []
parm_list = []
input_list = []
f1_list = []
roc_auc_list = []
class_num = 3
global img_test_acc
img_test_acc = 0
global candi
candi = 0
num_save = 0
time_count = 0
trtime_list = []
tetime_list = []
total_time_list = []
if __name__ == "__main__":
    ucr = fetch_ucr_dataset('ArrowHead', use_cache=True, data_home=None, return_X_y=False)  # 5 class
    ucr_train = ucr.data_train  # 390, 176
    ucr_target_train = ucr.target_train
    ucr_test = ucr.data_test
    ucr_target_test = ucr.target_test
    ucr_train_re = resize(ucr_train, ((36, 251, 1)))
    ucr_test_re = resize(ucr_test, ((175, 251, 1)))
    trencoder = LabelEncoder()
    trencoder.fit(ucr_target_train)
    ucr_target_train = trencoder.transform(ucr_target_train)
    teencoder = LabelEncoder()
    teencoder.fit(ucr_target_test)
    ucr_target_test = teencoder.transform((ucr_target_test))
    #class_num = 60
    train_size = len(ucr_train_re)
    test_size = len(ucr_test_re)
    input_size = 64

    tran = Image_coding(decompose=True)
    train_rp, train_rerp, train_trrp, train_gas, train_regas, train_trgas, train_gad, train_regad, train_trgad, train_mk, train_remk, train_trmk, train_ctwav = tran(
        ucr_train_re)  # wavelet 추가?
    test_rp, test_rerp, test_trrp, test_gas, test_regas, test_trgas, test_gad, test_regad, test_trgad, test_mk, test_remk, test_trmk, test_ctwav = tran(
        ucr_test_re)


    train_rp = resize(train_rp, ((train_size, input_size, input_size)))
    test_rp = resize(test_rp, ((test_size, input_size, input_size)))
    train_trrp = resize(train_trrp, ((train_size, input_size, input_size)))
    test_trrp = resize(test_trrp, ((test_size, input_size, input_size)))
    train_rerp = resize(train_rerp, ((train_size, input_size, input_size)))
    test_rerp = resize(test_rerp, ((test_size, input_size, input_size)))

    train_gad = resize(train_gad, ((train_size, input_size, input_size)))
    test_gad = resize(test_gad, ((test_size, input_size, input_size)))
    train_trgad = resize(train_trgad, ((train_size, input_size, input_size)))
    test_trgad = resize(test_trgad, ((test_size, input_size, input_size)))
    train_regad = resize(train_regad, ((train_size, input_size, input_size)))
    test_regad = resize(test_regad, ((test_size, input_size, input_size)))

    train_gas = resize(train_gas, ((train_size, input_size, input_size)))
    test_gas = resize(test_gas, ((test_size, input_size, input_size)))
    train_trgas = resize(train_trgas, ((train_size, input_size, input_size)))
    test_trgas = resize(test_trgas, ((test_size, input_size, input_size)))
    train_regas = resize(train_regas, ((train_size, input_size, input_size)))
    test_regas = resize(test_regas, ((test_size, input_size, input_size)))

    train_mk = resize(train_mk, ((train_size, input_size, input_size)))
    test_mk = resize(test_mk, ((test_size, input_size, input_size)))
    train_trmk = resize(train_trmk, ((train_size, input_size, input_size)))
    test_trmk = resize(test_trmk, ((test_size, input_size, input_size)))
    train_remk = resize(train_remk, ((train_size, input_size, input_size)))
    test_remk = resize(test_remk, ((test_size, input_size, input_size)))

    train_ctwav = resize(train_ctwav, ((train_size, input_size, input_size)))
    test_ctwav = resize(test_ctwav, ((test_size, input_size, input_size)))

    train_dict = {"rp": train_rp, "rerp": train_rerp, "trrp": train_trrp, "gas": train_gas, "regas": train_regas,
                  "trgas": train_trgas, "gad": train_gad, "regad": train_regad, "trgad": train_trgad, "mk": train_mk,
                  "remk": train_remk, "trmk": train_trmk, "ctwav": train_ctwav}
    test_dict = {"rp": test_rp, "rerp": test_rerp, "trrp": test_trrp, "gas": test_gas, "regas": test_regas,
                 "trgas": test_trgas, "gad": test_gad, "regad": test_regad, "trgad": test_trgad, "mk": test_mk,
                 "remk": test_remk, "trmk": test_trmk, "ctwav": test_ctwav}
    coding_name = ["rp", "rerp", "trrp", "gas", "regas", "trgas", "gad", "regad", "trgad", "mk", "remk", "trmk",
                   "ctwav"]

    for i in coding_name:
        train_dict[i] = np.expand_dims(train_dict[i], axis=1)
        test_dict[i] = np.expand_dims(test_dict[i], axis=1)

    PATH = "/home/sin/PycharmProjects/please/save/ElectricDevices/"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "1"
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("current cuda device:", torch.cuda.current_device())
    print("count of using gpus:", torch.cuda.device_count() )


    mlp = Babroad()
    default_config = mlp.configspace.get_default_configuration()

    # Define our environment variables
    scenario = Scenario(mlp.configspace, n_trials=10)

    # We also want to include our default configuration in the initial design
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        additional_configs=[default_config],
    )

    # We define the prior acquisition function, which conduct the optimization using priors over the optimum
    acquisition_function = PriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(scenario),
        decay_beta=scenario.n_trials / 10,  # Proven solid value
    )

    # We only want one config call (use only one seed in this example)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = HyperparameterOptimizationFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(default_config)
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"choosed cost: {incumbent_cost}")
    all_trtime = 0
    all_tetime = 0
    all_total_time = 0
    print(trtime_list)
    print(time_count)
    for i in range(time_count):
        all_trtime = all_trtime+ trtime_list[i]
        all_tetime = all_tetime+tetime_list[i]
        all_total_time = all_total_time+total_time_list[i]
    avg_trtime = all_trtime/time_count
    avg_tetime = all_tetime/time_count
    avg_total_time = all_total_time/time_count
    print("avg_train_time : ", avg_trtime)
    print("avg_test_time : ", avg_tetime)
    print("avg_all_time : ", avg_total_time)
