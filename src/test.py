
# data = []
# for x in range(57):
#     data.append(x)
# num_rounds, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1, class_occurances_correct, class_occurances_total = [], [], [], [], [], [], [], [], []

# print("data.shape", len(data))
# #class_occurances_total = [0] * 25
# #class_occurances_correct = [0] * 25
# for x in range(len(data)):
#     print(data[x])
#     num_rounds.append(data[x])
#     local_ae_loss.append(data[x])
#     train_loss.append(data[x])
#     train_accuracy.append(data[x])
#     test_loss.append(data[x])
#     test_accuracy.append(data[x])
#     test_f1.append(data[x])
#     class_occurances_correct.append(data[7:32])
#     class_occurances_total.append(data[32:])

# print("class_occurances_correct", len(class_occurances_correct), len(class_occurances_correct[0]))
# print("class_occurances_total", len(class_occurances_total), len(class_occurances_total[0]))

# data = []
# for x in range(57):
#     data.append(x)

# #print("data", len(data), data)


# class_occurances_correct = []
# class_occurances_total = []
# class_occurances_correct.append(data[7:32])
# class_occurances_total.append(data[32:])


# print("\nclass_occurances_correct", len(class_occurances_correct), len(class_occurances_correct[0]))

# print("class_occurances_total", len(class_occurances_total), len(class_occurances_total[0]))



import matplotlib.pyplot as plt

# Sample data for the table
data = [
    ['Name', 'Age', 'Gender'],
    ['Alice', 25, 'Female'],
    ['Bob', 30, 'Male'],
    ['Charlie', 28, 'Male'],
    ['Diana', 35, 'Female']
]

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
