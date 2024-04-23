import torch.utils.data as Data
import numpy as np
import torch

class Progress():
    """
    Early Stopping monitoring via training progress as proposed by:
    Prechelt, Lutz. Early Stopping—But When? Neural Networks: Tricks of the Trade (2012): 53-67.

    Args:
        strip (int): sequence length of training observed
        threshold (float): lower threshold on progress
    """
    def __init__(self, strip=5, threshold=0.01):
        self.strip = strip
        self.E = np.ones(strip)
        self.t = 0
        self.valid = False
        self.threshold = threshold

    def progress(self):
        """
        Compute the progress based on recent training errors.

        Returns:
            float: Progress value indicating how much the mean training error deviates from min over the strip
        """
        return 1000 * ((self.E.mean() / self.E.min()) - 1.)

    def stop(self):
        if self.valid == False:
            return False
        r = (self.progress() < self.threshold)
        return r

    def update(self, e):
        """
        Update the Progress object with a new training error.

        Args:
            e (float): Training error for the current iteration.

        Returns:
            bool: True if training should stop based on the updated progress, False otherwise.
        """
        self.E[np.mod(self.t, self.strip)] = e
        self.t += 1
        if self.t >= self.strip:
            self.valid = True
        return self.stop()


def fit_certified_MN(model, crit, optim, reg_strength, x,y, x_test, y_test, mono_feature=1, num_epochs=2000):
    '''
    Training procedure with monotonicity regularization (penalty) as described in
    Liu, Han et al. 2020 – "Certified Monotonic Neural Networks"
    :param model: MLP model
    :param crit:
    :param optim:
    :param x: train data
    :param y: train labels
    :param x_test: validation data
    :param y_test: validation labels (function restores state of lowest validation error)
    :param num_epochs:
    :return: None (model called by reference)
    '''
    feature_num = x.shape[1]
    # mono_feature = 1
    val_loss = 1000
    threshold = 1e-4
    P = Progress(5, threshold=threshold)

    data_train = Data.TensorDataset(x, y)
    data_train_loader = Data.DataLoader(
        dataset=data_train,
        batch_size=len(data_train),
        shuffle=True,
        # num_workers=2,  #
    )
    history = []

    for i in range(num_epochs):
        ### training
        model.train()
        loss_avg = 0.
        grad_loss_avg = 0.
        batch_idx = 0
        for X, y in iter(data_train_loader):
            batch_idx += 1
            # X, y = X.cuda(), y.cuda()
            optim.zero_grad()
            X.requires_grad = True
            out = model(X[:, :mono_feature], X[:, mono_feature:])
            loss = crit(out, y)

            in_list, out_list = model.reg_forward(feature_num=feature_num, mono_num=mono_feature, num=1024)
            reg_loss = generate_regularizer(in_list, out_list)

            loss_total = loss + reg_strength * reg_loss
            stop = P.update(loss_total.item())
            if (stop):
                print("Early Stopping.")
                return

            loss_avg += loss.detach().numpy()
            # loss_avg += loss.detach().cpu().numpy()
            grad_loss_avg += reg_loss

            loss_total.backward()
            optim.step()
        history.append(loss_total/ (len(data_train_loader.dataset)/data_train_loader.batch_size))
        # scheduler.step()
        if (i + 1) % 100 == 0:
            # print(f"loss: {loss}, reg_loss: {reg_loss}, total_loss: {loss_total}")
            print(f'\t\tEpoch [{i+1}/{num_epochs}], Loss: {loss_avg / (len(data_train_loader.dataset)/data_train_loader.batch_size)}, Grad Loss: {grad_loss_avg / batch_idx}')
    return np.array([i.detach().numpy() for i in history])
    #     # section commented out as we dont optimize on the test set
    #     model.eval()
    #     val_loss_cur = val(model, x_test, y_test, crit, mono_feature)
    #     # test_loss_cur = test()
    #     if (val_loss > val_loss_cur):
    #     #     test_loss = test_loss_cur
    #         val_loss = val_loss_cur
    #         torch.save(model.state_dict(), 'ressources/net_MA.pth')
    #
    #     # print('best', val_loss)
    # model.load_state_dict(torch.load('ressources/net_MA.pth'))

def fit_certified_MN_val(model, crit, optim, reg_strength, x,y, val_split=.2, patience=100, mono_feature=1, num_epochs=2000):
    '''
    Training procedure with monotonicity regularization (penalty) as described in
    Liu, Han et al. 2020 – "Certified Monotonic Neural Networks"
    :param model: MLP model
    :param crit:
    :param optim:
    :param x: train data
    :param y: train labels
    :param x_test: validation data
    :param y_test: validation labels (function restores state of lowest validation error)
    :param num_epochs:
    :return: None (model called by reference)
    '''
    end = int(len(x) - len(x) * val_split)
    x_val = x[end:]
    y_val = y[end:]
    x = x[:end]
    y = y[:end]
    feature_num = x.shape[1]
    val_loss = 1000

    data_train = Data.TensorDataset(x, y)
    data_train_loader = Data.DataLoader(
        dataset=data_train,
        batch_size=len(data_train),
        shuffle=True,
    )
    history = []
    best_epoch = 0

    for i in range(num_epochs):
        ### training
        model.train()
        loss_avg = 0.
        grad_loss_avg = 0.
        batch_idx = 0
        for X, y in iter(data_train_loader):
            batch_idx += 1
            # X, y = X.cuda(), y.cuda()
            optim.zero_grad()
            X.requires_grad = True
            out = model(X[:, :mono_feature], X[:, mono_feature:])
            loss = crit(out, y)

            in_list, out_list = model.reg_forward(feature_num=feature_num, mono_num=mono_feature, num=1024)
            reg_loss = generate_regularizer(in_list, out_list)
            loss_total = loss + reg_strength * reg_loss

            loss_avg += loss.detach().numpy()
            grad_loss_avg += reg_loss

            loss_total.backward()
            optim.step()
        history.append(loss_total/ (len(data_train_loader.dataset)/data_train_loader.batch_size))
        if (i + 1) % 100 == 0:
            print(f"loss: {loss}, reg_loss: {reg_loss}, total_loss: {loss_total}")
            print(f'\t\tEpoch [{i+1}/{num_epochs}], Loss: {loss_avg / (len(data_train_loader.dataset)/data_train_loader.batch_size)}, Grad Loss: {grad_loss_avg / batch_idx}')
        model.eval()
        val_loss_cur = val(model, x_val, y_val, crit, mono_feature, reg_strength)
        if (i == 0):
            val_loss = val_loss_cur
            torch.save(model.state_dict(), 'ressources/net_MA.pth')
        if (val_loss > val_loss_cur):
            val_loss = val_loss_cur
            best_epoch = i
            torch.save(model.state_dict(), 'ressources/net_MA.pth')
        if (i - best_epoch > patience):
            model.load_state_dict(torch.load('ressources/net_MA.pth'))
            print(f"Early stopped at epoch{i}. Loaded weights from epoch {best_epoch} with best val_loss: {val_loss}")
            return np.array([i.detach().numpy() for i in history])
        # print('best', val_loss)
    model.load_state_dict(torch.load('ressources/net_MA.pth'))
    return np.array([i.detach().numpy() for i in history])

def generate_regularizer(in_list, out_list):
    length = len(in_list)
    reg_loss = 0.
    min_derivative = 0.0
    for i in range(length):
        xx = in_list[i]
        yy = out_list[i]
        for j in range(yy.shape[1]):
            grad_input = torch.autograd.grad(torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True)[0]
            grad_input_neg = -grad_input
            grad_input_neg += .2
            grad_input_neg[grad_input_neg < 0.] = 0.
            if min_derivative < torch.max(grad_input_neg**2):
                min_derivative = torch.max(grad_input_neg**2)
    reg_loss = min_derivative
    return reg_loss

# def test():
#     ### test:
#     out = net(X_test[:, :mono_feature], X_test[:, mono_feature:])
#     t_loss = criterion(out, y_test)
#     print('test loss:', t_loss.item())
#     # out[out>0.] = 1.
#     # out[out<0.] = 0.
#     # print('test accuracy:', torch.sum(out==y_test)/float(y_test.numel()))
#
#     # return torch.sum(out==y_test)/float(y_test.numel())
#     return t_loss

def val(net, X_val, y_val, criterion, mono_feature, reg_strength):
    ### val:
    out = net(X_val[:, :mono_feature], X_val[:, mono_feature:])
    v_loss = criterion(out, y_val)
    in_list, out_list = net.reg_forward(feature_num=X_val.shape[1], mono_num=mono_feature, num=1024)
    v_reg_loss = generate_regularizer(in_list, out_list)
    v_loss_total = v_loss + reg_strength * v_reg_loss
    # print('val loss:', v_loss.item())
    # out[out>0.] = 1.
    # out[out<0.] = 0.
    # print('val accuracy:', torch.sum(out==y_val)/float(y_val.numel()))

    # return torch.sum(out==y_val)/float(y_val.numel())
    return v_loss_total