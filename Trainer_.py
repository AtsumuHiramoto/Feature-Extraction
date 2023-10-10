import torch
import matplotlib.pyplot as plt
import os

def Train(model, train_dataset, test_dataset, decode_pe_flag = 1, cfg=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = model.to(device)
    print(model)
    lr = cfg["train"]["learningrate"]
    batch_size = cfg["train"]["batch_size"]
    if cfg["train"]["lossType"]=="mse":
        criterion = torch.nn.MSELoss()
    if cfg["train"]["optimizerType"]=="adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    num_epoch = cfg["train"]["epoch"]
    test_epoch = cfg["train"]["test_epoch"]
    train_loss_list = []
    test_loss_epoch = []
    test_loss_list = []
    for epoch in range(num_epoch):
        print("{} epoch".format(epoch))
        # import ipdb; ipdb.set_trace()
        for traindata in train_dataloader:
            inputs = traindata["Tactile"].to(torch.float32)
            if "Coordinates_Tactile" in cfg["data"]["inputType"]:
                cog = traindata["CoG_Tactile"]
                if decode_pe_flag:
                    cog_pe = traindata["CoG_Tactile_pe"]
                # import ipdb; ipdb.set_trace()
                optimizer.zero_grad()
                if decode_pe_flag:
                    features, outputs = model(inputs, cog, cog_pe)
                else:
                    features, outputs = model(inputs, cog)
            else:
                optimizer.zero_grad()
                features, outputs = model(inputs)
            train_loss = criterion(outputs, inputs)
            train_loss.backward()
            optimizer.step()
        print("train loss: {}".format(train_loss))
        train_loss_list.append(train_loss.cpu().detach().numpy())
        # import GPUtil
        # import ipdb; ipdb.set_trace()
        # del inputs, cog, features
        # torch.cuda.empty_cache()
        # import ipdb; ipdb.set_trace()

        if epoch%test_epoch==0 or (epoch+1)==num_epoch:
            for testdata in test_dataloader:
                # inputs, labels = inputs.to(device), labels.to(device)
                inputs = testdata["Tactile"].to(torch.float32)
                if "Coordinates_Tactile" in cfg["data"]["inputType"]:
                    cog = traindata["CoG_Tactile"]
                    optimizer.zero_grad()
                    if decode_pe_flag:
                        features, outputs = model(inputs, cog, cog_pe)
                    else:
                        features, outputs = model(inputs, cog)
                else:
                    optimizer.zero_grad()
                    features, outputs = model(inputs)
                loss = criterion(outputs, inputs)
            print("test loss: {}".format(loss))
            test_loss_list.append(loss.cpu().detach().numpy())
            test_loss_epoch.append(epoch)
    
    plt.plot(range(num_epoch), train_loss_list, label="train")
    plt.plot(test_loss_epoch, test_loss_list, label="test")
    plt.title("MSE Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # plt.show()
    count = 0
    while os.path.isfile("./loss/{}_{}.jpg".format(num_epoch, count))==True:
        count = count+1
        # import ipdb; ipdb.set_trace()
    plt.savefig("./loss/{}_{}.jpg".format(num_epoch, count))
    
    return