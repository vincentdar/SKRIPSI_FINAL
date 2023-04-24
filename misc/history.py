import matplotlib.pyplot as plt

with open('history.txt') as f:    
    lines = f.readlines()

epoch = 1
loss_lst = []
acc_lst = []
val_loss_lst = []
val_acc_lst = []
for line in lines:    
    split = line.split()
    if 'val_loss:' in split:        

        # Get
        loss = float(split[7])
        acc = float(split[10])
        val_loss = float(split[13])
        val_acc = float(split[16])

        
        print("Epoch:", epoch, "Loss:", loss, "Accuracy:", acc, "Val_loss", val_loss, "Val_accuracy", val_acc)
                

        loss_lst.append(loss)
        acc_lst.append(acc)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        epoch += 1

epochs = range(len(loss_lst))

title = "transfer_mobilenet_cnnlstm_tfrecord_100_epoch"
# Plot loss
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_lst, label="training_loss")
plt.plot(epochs, val_loss_lst, label="val_loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(bottom=0.0)
plt.grid(True)
plt.legend(loc = "lower left")

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_lst, label="training_accuracy")
plt.plot(epochs, val_acc_lst, label="val_accuracy")
plt.title("accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend(loc = "lower left")
plt.savefig(title + '_accuracy.png')
        

    


    