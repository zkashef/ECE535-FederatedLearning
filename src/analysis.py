import numpy as np
import os.path
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from pathlib import Path

METRIC = "f1"  # "accuracy", "f1"
METRIC_COLUMNS = {
    "accuracy": 5,
    "f1": 6,
}
METRIC_INDEX = METRIC_COLUMNS[METRIC]
DATASETS = [
    "Opp",
    "mHealth",
    "UR_Fall",
]

ROUND_CUT = {
    "Opp": {"single": 50,
            "cross": 50},
    "mHealth": {"single": 10,
                "cross": 25},
    "UR_Fall": {"single": 100,
                "cross": 100},
}

aes = {
    "Opp": ["dccae"],
    "mHealth": ["split_ae"],
    "UR_Fall": ["split_ae"],
}
ae_print = {
    "split_ae": "SplitAE",
    "dccae": "DCCAE",
}
combo = {
    "Opp": [("acce", "gyro")],
    "mHealth": [("acce", "gyro"), ("acce", "mage"), ("gyro", "mage")],
    "UR_Fall": [("acce", "depth"), ("rgb", "depth")],
}

cross_selected = {
    "Opp": {"acce_gyro": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        #   "client_ABB_label_B_test_A",
        #   "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        # "client_ABA_label_A_test_B",
        #   "client_ABB_label_A_test_B",
        "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ]
    },
    "mHealth": {"acce_gyro": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        #   "client_ABA_label_B_test_A",
        "client_ABB_label_B_test_A",
        # "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        # "client_ABA_label_A_test_B",
        "client_ABB_label_A_test_B",
        #   "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "acce_mage": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        #   "client_ABA_label_B_test_A",
        "client_ABB_label_B_test_A",
        #   "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABA_label_A_test_B",
        #   "client_ABB_label_A_test_B",
        #   "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "gyro_mage": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        #   "client_ABB_label_B_test_A",
        #   "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        # "client_ABA_label_A_test_B",
        "client_ABB_label_A_test_B",
        #   "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ]},
    "UR_Fall": {"acce_depth": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        #    "client_ABB_label_B_test_A",
        #    "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        #    "client_ABA_label_A_test_B",
        "client_ABB_label_A_test_B",
        #    "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "rgb_depth": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        #   "client_ABA_label_B_test_A",
        #   "client_ABB_label_B_test_A",
        "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",

        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        #   "client_ABA_label_A_test_B",
        #   "client_ABB_label_A_test_B",
        "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",

    ]}
}

modality_print = {
    "acce": "Acce",
    "gyro": "Gyro",
    "mage": "Mag",
    "rgb": "RGB",
    "depth": "Depth",
}
N_REPS = 1

CB_color_cycle = ['#ff7f00', '#377eb8',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', '#4daf4a']

            


def single_multi_modality_comparison():
    # Iterate over each dataset
    for dataset in DATASETS:
        # Iterate over each autoencoder model for the current dataset
        for ae in aes[dataset]:
            # Iterate over each pair of modalities for the current dataset
            for modalities in combo[dataset]:
                # Create a new matplotlib figure and axes
                plt.figure()
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
                # Setting Y limit and fonts for the plot
                ax.set_ylim([0.0, 1.1])
                plt.xticks(fontsize="large")
                plt.yticks(fontsize="large")
                # Define Schemes and Legend Labels 
                schemes = {"client_A_label_A_test_A": "A30_B0_AB0_label_A_test_A",
                           "client_B_label_B_test_B": "A0_B30_AB0_label_B_test_B",
                           "client_AB_label_AB_test_A": "A0_B0_AB30_label_AB_test_A",
                           "client_AB_label_AB_test_B": "A0_B0_AB30_label_AB_test_B", }
                legends = {"client_A_label_A_test_A": "$\mathregular{UmFL_A}$",
                           "client_B_label_B_test_B": "$\mathregular{UmFL_B}$",
                           "client_AB_label_AB_test_A": "$\mathregular{MmFL_{AB}-L_{AB}-T_A}$",
                           "client_AB_label_AB_test_B": "$\mathregular{MmFL_{AB}-L_{AB}-T_B}$", }
                # Iterate over each scheme
                for k in schemes:
                    # Determine color and linestyle based on scheme
                    color_idx = 0 if k[-1] == "A" else 1
                    linestyle = ("-" if color_idx == 0 else "dashdot")if k[8] != "_" else ("--" if color_idx == 0 else (0, (5, 10)))
                    test_acc = []
                    # Iterate over each repetition
                    for rep in range(N_REPS):
                        # Construct file path to read results
                        rep_file = os.path.join("results", dataset.lower(), ae, f"{modalities[0]}_{modalities[1]}", schemes[k], "results.txt")
                        # Load results data from file
                        data = np.loadtxt(rep_file, delimiter=",")
                        x_all = data[:, 0]
                        idxs_round_cut = x_all <= ROUND_CUT[dataset]["single"]
                        x = x_all[idxs_round_cut]
                        y_rep = data[idxs_round_cut, METRIC_INDEX]
                        # Replace any zero performance values with np.nan (will be ignored when averaging)
                        y_rep[y_rep == 0.0] = np.nan
                        test_acc.append(y_rep)
                    # Calculate mean and SEM performance across repetitions
                    y = np.nanmean(np.array(test_acc), axis=0)
                    se = stats.sem(np.array(test_acc), nan_policy="omit")
                    # Plot line with the performance of the scheme
                    ax.plot(x, y, color=CB_color_cycle[color_idx], linestyle=linestyle, label=legends[k])
                    # Fill area between lines to represent standard deviation
                    ax.fill_between(x, y-se, y+se, color=CB_color_cycle[color_idx], alpha=0.3)
                # Set labels and title for the plot
                ax.set_xlabel("Communication rounds", fontsize="x-large")
                metric_label = "accuracy" if METRIC == "accuracy" else "$\mathregular{F_1}$"
                ax.set_ylabel(f"Test {metric_label}", fontsize="x-large")
                dataset_print = dataset.replace("_", " ")
                ax.set_title(f"{dataset_print}, {ae_print[ae]}, A: {modality_print[modalities[0]]}, B: {modality_print[modalities[1]]}")
                # Set legend for the plot
                ax.legend(loc="lower right", fontsize="x-large")
                # Save the plot
                Path("plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"plots/single_multi_modality_comparison_{dataset}_{ae}_{modalities[0]}_{modalities[1]}.pdf",
                            bbox_inches="tight")


def cross_modality_comparison():
    for dataset in DATASETS:
        for ae in aes[dataset]:
            for modalities in combo[dataset]:
                plt.figure()
                fig, axes = plt.subplots(
                    nrows=1, ncols=2, figsize=(10, 3.7))

                schemes_test_A = {
                    "client_A_label_A_test_A": (0, "A30_B0_AB0_label_A_test_A"),
                    "client_AB_label_B_test_A": (1, "A0_B0_AB30_label_B_test_A"),
                    "client_ABA_label_B_test_A": (2, "A10_B0_AB30_label_B_test_A"),
                    "client_ABB_label_B_test_A": (3, "A0_B10_AB30_label_B_test_A"),
                    "client_ABAB_label_B_test_A": (4, "A10_B10_AB30_label_B_test_A"),
                    "ablation_label_B_test_A": (5, "A30_B30_AB0_label_B_test_A"),
                }
                schemes_test_B = {
                    "client_B_label_B_test_B": (0, "A0_B30_AB0_label_B_test_B"),
                    "client_AB_label_A_test_B": (1, "A0_B0_AB30_label_A_test_B"),
                    "client_ABB_label_A_test_B": (2, "A0_B10_AB30_label_A_test_B"),
                    "client_ABA_label_A_test_B": (3, "A10_B0_AB30_label_A_test_B"),
                    "client_ABAB_label_A_test_B": (4, "A10_B10_AB30_label_A_test_B"),
                    "ablation_label_A_test_B": (5, "A30_B30_AB0_label_A_test_B"),
                }
                legends = {
                    "client_A_label_A_test_A": "$\mathregular{UmFL_A}$",
                    "client_AB_label_B_test_A": "$\mathregular{MmFL_{AB}-L_B-T_A}$",
                    "client_ABA_label_B_test_A": "$\mathregular{MmFL_{ABA}-L_B-T_A}$",
                    "client_ABB_label_B_test_A": "$\mathregular{MmFL_{ABB}-L_B-T_A}$",
                    "client_ABAB_label_B_test_A": "$\mathregular{MmFL_{ABAB}-L_B-T_A}$",
                    "ablation_label_B_test_A": "$\mathregular{Abl-L_B-T_A}$",
                    "client_B_label_B_test_B": "$\mathregular{UmFL_B}$",
                    "client_AB_label_A_test_B": "$\mathregular{MmFL_{AB}-L_A-T_B}$",
                    "client_ABB_label_A_test_B": "$\mathregular{MmFL_{ABB}-L_A-T_B}$",
                    "client_ABA_label_A_test_B": "$\mathregular{MmFL_{ABA}-L_A-T_B}$",
                    "client_ABAB_label_A_test_B": "$\mathregular{MmFL_{ABAB}-L_A-T_B}$",
                    "ablation_label_A_test_B": "$\mathregular{Abl-L_A-T_B}$",
                }

                groups = (schemes_test_A, schemes_test_B)
                for col, schemes in enumerate(groups):
                    ax = axes[col]
                    ax.set_ylim([0.0, 1.1])
                    plt.xticks(fontsize="large")
                    plt.yticks(fontsize="large")
                    for k in schemes:
                        if k not in cross_selected[dataset][f"{modalities[0]}_{modalities[1]}"]:
                            continue
                        color_idx = schemes[k][0]
                        linestyle = (
                            "-" if color_idx == 1 else "dashdot") if color_idx != 0 and color_idx != 5 else "--" if color_idx == 0 else "dotted"
                        test_acc = []
                        for rep in range(N_REPS):
                            if "ablation" not in k:
                                rep_file = os.path.join(
                                    "results", dataset.lower(), ae, f"{modalities[0]}_{modalities[1]}", schemes[k][1], "results.txt")
                            else:
                                rep_file = os.path.join(
                                    "results", dataset.lower(), "ablation", f"{modalities[0]}_{modalities[1]}", schemes[k][1], "results.txt")
                            data = np.loadtxt(rep_file, delimiter=",")
                            x_all = data[:, 0]
                            idxs_round_cut = x_all <= ROUND_CUT[dataset]["cross"]
                            x = x_all[idxs_round_cut]
                            y_rep = data[idxs_round_cut, METRIC_INDEX]
                            y_rep[y_rep == 0.0] = np.nan
                            test_acc.append(y_rep)
                        y = np.nanmean(np.array(test_acc), axis=0)
                        ax.plot(
                            x, y, color=CB_color_cycle[color_idx], linestyle=linestyle, label=legends[k])
                        se = stats.sem(np.array(test_acc), nan_policy="omit")
                        ax.fill_between(x, y-se, y+se,
                                        color=CB_color_cycle[color_idx], alpha=0.3)
                    ax.set_xlabel("Communication rounds", fontsize="x-large")
                    metric_label = "accuracy" if METRIC == "accuracy" else "$\mathregular{F_1}$"
                    ax.set_ylabel(
                        f"Test {metric_label}", fontsize="x-large")
                    dataset_print = dataset.replace("_", " ")
                    ax.set_title(
                        f"{dataset_print}, {ae_print[ae]}, A: {modality_print[modalities[0]]}, B: {modality_print[modalities[1]]}")
                    ax.legend(loc="lower right", fontsize="x-large")
                Path("plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"plots/cross_modality_comparison_{dataset}_{ae}_{modalities[0]}_{modalities[1]}.pdf",
                            bbox_inches="tight")




# In server.py, y contains the labels (classes)
# equals is an array of booleans representing if the model correctly guessed
# From those two you should be able to count how many guesses are correct per class
def per_class_accuracy():
    # add all files from path to list
    repo_path = '/Users/zach/Desktop/School/Fall2023/ECE535/ECE535-FederatedLearning/results/'
    dataset = ['mhealth/', 'opp/', 'ur_fall/']

    type_mhealth = ['split_ae/', 'ablation/']
    type_opp = ['dccae/', 'ablation/']
    type_urfall = ['split_ae/', 'ablation/']
    
    modality_mhealth = ['acce_gyro/', 'acce_mage/', 'gyro_mage/']
    modality_opp = ['acce_gyro/']
    modality_urfall = ['acce_depth/', 'acce_rgb/', 'rgb_depth/']
    
    for y in dataset:
        print(y)
        if y == 'mhealth/':
            current_type = type_mhealth
            current_modality = modality_mhealth
        elif y == 'opp/':
            current_type = type_opp
            current_modality = modality_opp
        elif y == 'ur_fall/':
            current_type = type_urfall
            current_modality = modality_urfall
        
        for x in current_type:
            print(x)
            for z in current_modality:
                print(z)
                for file in os.listdir(repo_path+y+x+z):
                    print("file: ", repo_path+y+x+z+file)
                    if file != '.DS_Store':
                        #per_class_accuracy_plot(repo_path, y, x, z, file)
                        per_class_accuracy_chart(repo_path, y, x, z, file)
                        return

                

                    

def per_class_accuracy_chart(rep_path, dataset, datatype, modality, file):
    # Construct file path to read results
    rep_file = os.path.join(rep_path, dataset, datatype, modality, file, "results.txt")
    # Load results data from file
    data = np.loadtxt(rep_file, delimiter=",")
    # #t+1, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1, *class_accuracy
    num_rounds, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1, class_occurances_correct, class_occurances_total = [], [], [], [], [], [], [], [], []
    
    # print("data.shape", data.shape)
    # print("data len", len(data), len(data[0]))
    for x in range(len(data)):
        #print(x, data[x])
        num_rounds.append(data[x][0])
        local_ae_loss.append(data[x][1])
        train_loss.append(data[x][2])
        train_accuracy.append(data[x][3])
        test_loss.append(data[x][4])
        test_accuracy.append(data[x][5])
        test_f1.append(data[x][6])
        class_occurances_correct.append(data[x][7:32])
        class_occurances_total.append(data[x][32:])
    

    # indentity when the last element of class_occurances_total is 0 and class_occurances_correct is 0
    # delete each empty element from class_occurances_total and class_occurances_correct
    for y in range(25):
        for x in range(len(class_occurances_total)):
            if class_occurances_total[x][-1] == 0:
                # delete last element of each np array
                class_occurances_total[x] = np.delete(class_occurances_total[x], -1)
                class_occurances_correct[x] = np.delete(class_occurances_correct[x], -1)

    # check that the same amont of empty classes were deleted from both class_occurances_correct and class_occurances_total
    #print("check length of correct & total: ", len(class_occurances_correct) == len(class_occurances_total))
    data = [] * len(num_rounds)

    print(data)
    print(class_occurances_correct[0])
    print(class_occurances_total[0])
    for x in range(len(num_rounds)):
        print(num_rounds[x])

        data[x].append(num_rounds[x])
        for y in range(len(class_occurances_correct[x])):
            data[x].append(class_occurances_correct[x][y], class_occurances_total[x][y])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide the axes to display only the table
    ax.axis('off')

    # Create the table and add it to the plot
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=None)

    # Modify table properties (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust the table size

    plt.title('Sample Table')  # Optional: Set a title for the plot
    plt.show()


def per_class_accuracy_plot(rep_path, dataset, datatype, modality, file):
    # Construct file path to read results
    rep_file = os.path.join(rep_path, dataset, datatype, modality, file, "results.txt")
    # Load results data from file
    data = np.loadtxt(rep_file, delimiter=",")
    # #t+1, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1, *class_accuracy
    num_rounds, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1, class_occurances_correct, class_occurances_total = [], [], [], [], [], [], [], [], []
    
    # print("data.shape", data.shape)
    # print("data len", len(data), len(data[0]))
    for x in range(len(data)):
        #print(x, data[x])
        num_rounds.append(data[x][0])
        local_ae_loss.append(data[x][1])
        train_loss.append(data[x][2])
        train_accuracy.append(data[x][3])
        test_loss.append(data[x][4])
        test_accuracy.append(data[x][5])
        test_f1.append(data[x][6])
        class_occurances_correct.append(data[x][7:32])
        class_occurances_total.append(data[x][32:])
    

    # indentity when the last element of class_occurances_total is 0 and class_occurances_correct is 0
    # delete each empty element from class_occurances_total and class_occurances_correct
    for y in range(25):
        for x in range(len(class_occurances_total)):
            if class_occurances_total[x][-1] == 0:
                # delete last element of each np array
                class_occurances_total[x] = np.delete(class_occurances_total[x], -1)
                class_occurances_correct[x] = np.delete(class_occurances_correct[x], -1)

    # check that the same amont of empty classes were deleted from both class_occurances_correct and class_occurances_total
    #print("check length of correct & total: ", len(class_occurances_correct) == len(class_occurances_total))

    # reorganize class_accuracy into individual lists for each class
    # class_accuracy is 2D array where each element is an array a different class
    class_accuracy = [[] for _ in range(len(class_occurances_correct[0]))]  # Create sublist for each classes
    for y in range(len(num_rounds)):
        for x in range(len(class_accuracy)):
            class_accuracy[x].append(class_occurances_correct[y][x] / class_occurances_total[y][x])

    #print("class_accuracy shape: ", len(class_accuracy), len(class_accuracy[0]))
    
    # class_accuracy vs round plot
    plt.figure()
    # create single plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
    # Setting Y limit and fonts for the plot
    ax.set_ylim([0.0, 1.1])
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    for x in range(len(class_accuracy)):
        random_color = np.random.rand(3,)  # Generating random RGB values
        ax.plot(num_rounds, class_accuracy[x], color=random_color, linestyle="-", label="Class "+str(x+1)) # random color
    ax.set_xlabel("Round", fontsize="x-large")
    ax.set_ylabel("Accuracy", fontsize="x-large")
    ax.set_title(f"{dataset}, {datatype}, {modality}, {file}")
    # Set legend for the plot
    ax.legend(loc="lower right", fontsize="small")
    # Display plot
    #plt.show()
    # save plot
    Path("plots").mkdir(parents=True, exist_ok=True)
        # Check if the directory exists, if not, create it
    if not os.path.exists("plots/per_class_analysis/"+dataset+datatype+modality):
        os.makedirs("plots/per_class_analysis/"+dataset+datatype+modality)
    plt.savefig(f"plots/per_class_analysis/"+dataset+datatype+modality+"class_accuracy_"+file+".pdf", bbox_inches="tight")


def main():
    #single_multi_modality_comparison()
    #cross_modality_comparison()
    per_class_accuracy()


if __name__ == "__main__":
    main()
