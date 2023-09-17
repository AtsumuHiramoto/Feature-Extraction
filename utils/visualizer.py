import matplotlib.pyplot as plt
import numpy as np

class Visualizer(object):
    def __init__(self) -> None:
        pass
    def save_loss_image(self, 
                        train_loss=None, 
                        test_loss=None, 
                        save_dir=None, 
                        model_name=None,
                        mode=""):
        assert mode=="" or mode=="log10", "mode={} is invalid".format(mode)
        label_list = []
        if train_loss is not None:
            label="train_loss"
            # import ipdb; ipdb.set_trace()
            if mode=="log10":
                train_loss = np.log10(train_loss)
            plt.plot(range(len(train_loss)), train_loss, "r", label=label)
            label_list.append
        if test_loss is not None:
            label="test_loss"
            if mode=="log10":
                test_loss = np.log10(test_loss)
            plt.plot(range(len(test_loss)), test_loss, "b", label=label)
            label_list.append(label)
        plt.legend()
        if save_dir is not None:
            if model_name is not None:
                file_name = "{}_loss".format(model_name)
                plt.title(file_name)
                plt.savefig(save_dir + file_name + "_" + mode + ".png")
            else:
                file_name = "loss"
                plt.title(file_name)
                plt.savefig(save_dir + file_name + "_" + mode + ".png")
        else:
            plt.show()

if __name__=="__main__":
    v = Visualizer()
    train_loss = [10,1,0.2,0.1,0.01, 0.007]
    test_loss = [20,3,2,1,0.3,0.1]
    save_dir = "./"
    model_name = "lstm"
    mode = "log10"
    v.save_loss_image(train_loss=train_loss,test_loss=test_loss,save_dir=save_dir,model_name=model_name,mode=mode)