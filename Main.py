import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from tensorflow.keras.utils import to_categorical



import Util.SCA_util as SCA_util
import Util.SCA_dataset as SCA_dataset
import Util.DL_model as DL_model

if __name__ == "__main__":

    file_root = ""  # dataset root
    dataset = './ASCAD_desync0.h5'
    leakage = 'HW'
    attack_model = 'MLP'
    metrics = 'all'
    profiling_traces = 50000
    model_size = 64
    epochs = [50]
    nb_traces_attacks = 5000
    nb_attacks = 10
    correct_key = 224
    attack_byte = 2
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), (
        key_profiling, key_attack) = SCA_dataset.load_ascad(dataset,
                                                            leakage_model=leakage,
                                                            profiling_traces=profiling_traces,
                                                            key_info=True)

    scaler = StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    # Select leakage model
    if leakage == 'ID':
        classes = 256
    else:
        classes = 9

    Y_profiling = np.concatenate((to_categorical(Y_profiling, num_classes=classes),
                                  np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
    Y_attack = np.concatenate(
        (to_categorical(Y_attack, num_classes=classes), np.ones((len(plt_attack), 1)), plt_attack), axis=1)

    Atk_ge = SCA_util.Attack(leakage, correct_key, nb_traces_attacks=0,
                             nb_attacks=0, attack_byte=attack_byte, shuffle=True,
                             output='prob_metric')
    Loss = SCA_util.custom_loss(leakage, Atk_ge)

    # Metric selection: ACC/AGE/key_rank
    metric = [SCA_util.acc_Metric(leakage)]
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage, X_profiling.shape[1], metric,
                                                       Loss.categorical_crossentropy,
                                                       model=attack_model, model_size=model_size)

    for epoch_idx, epoch in enumerate(epochs):
        if epoch == 'best':
            epoch = epoch_sota
        else:
            if epoch_idx == 0:
                epoch = int(epoch)
            else:
                epoch = int(epoch) - int(epochs[epoch_idx - 1])

        model_root = "model.h5"
        save_model = ModelCheckpoint("./Model/model_{epoch:03d}.h5")
        callbacks = [save_model]
        history = model.fit(x=X_profiling, y=Y_profiling, batch_size=batch_size, verbose=2, epochs=epoch,
                            callbacks=callbacks)

    NAs = []
    for i in range(1, 51):
        model_name = "./Model/model_" + str(i).zfill(3) + ".h5"
        model = load_model(model_name, compile=False)
        predictions = model.predict(X_attack)
        Atk_ge_age = SCA_util.Attack(leakage, correct_key, nb_traces_attacks=nb_traces_attacks,
                                     nb_attacks=nb_attacks, attack_byte=attack_byte, shuffle=True,
                                     output='rank')
        all_rank_evol = np.array(Atk_ge_age.perform_attacks(predictions, plt_attack))
        all_na = []
        for j in range(all_rank_evol.shape[0]):
            na = 5000
            rank_evol = all_rank_evol[j]
            for k in range(4999, -1, -1):
                if rank_evol[:, correct_key][k] == 0:
                    na = k
                else:
                    break
            all_na.append(na)
        print("Epoch_" + str(i) + ":" + str(np.mean(all_na)))
        NAs.append(np.mean(all_na))

    # NAs = [5000.0, 2480.5, 2474.6, 1806.6, 864.9, 1074.1, 899.9, 838.4, 1664.0, 1239.7, 912.5, 745.2, 2013.5, 585.9, 476.0, 892.5, 1055.1, 588.2, 981.3, 1159.6, 1014.5, 1062.0, 1263.2, 1578.6, 1081.7, 910.7, 1234.5, 924.5, 1431.4, 1159.5, 981.8, 1563.2, 897.0, 1505.4, 1118.3, 796.6, 1392.3, 1316.7, 1314.5, 1276.7, 1124.0, 1483.9, 1752.4, 1736.5, 1332.5, 1486.6, 1751.2, 1028.3, 1191.5, 1644.8]
    plt.plot(range(50), NAs)
    plt.xlabel("Epoch", fontdict={'weight': 'normal'})
    plt.ylabel("NA", fontdict={'weight': 'normal'})
    plt.savefig("NA.svg", format='svg')
    plt.show()
    print(NAs)
