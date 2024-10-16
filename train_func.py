import torch

'Computes and stores the average and current value.'
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

' network training function '
def train_net(model, device, loader, optimizer, loss_f,  batch_size):
    model.train()
    train_loss = AverageMeter()
    for it,data in enumerate(loader):
        imgs, labels = data
        labels = labels.squeeze(2)
        pred = model(imgs.to(device))
        loss = loss_f(pred, labels.to(device).long()) # Loss calculation
        train_loss.update(loss.item(), pred.size(0)) # Update the record
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(' Train_Loss: ' + str(round(train_loss.avg, 6)), end=" ")
    return train_loss.avg

' network validating function '
def val_net(model, device, loader, loss_f, batch_size):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for it, data in enumerate(loader):
            imgs, labels = data
            labels = labels.squeeze(2)
            pred = model(imgs.to(device))
            loss = loss_f(pred, labels.to(device).long()) # Loss calculation
            val_loss.update(loss.item(), pred.size(0)) # Update the record
    print(' Val_loss: ' + str(round(val_loss.avg, 6)))
    return val_loss.avg