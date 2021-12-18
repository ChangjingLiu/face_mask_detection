def train(model,conifg,device):
    # Main training function
    num_epochs = config['num_epochs']  # Maximum number of epochs
    loss_list = []
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Starting training....{}/{}'.format(epoch+1, num_epochs))
        loss_sub_list = []
        start = time.time()
        for images, targets in mask_loader:
            images = list(image.to(device) for image in images)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)

            # update optimizer and learning rate
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            #lr_scheduler.step()
        end = time.time()

        #print the loss of epoch and save
        epoch_loss = np.mean(loss_sub_list)
        if epoch_loss<min_loss:
            print("saving model")
            torch.save(model.state_dict(), '../checkpoint/model_0214.pth')
            min_loss = epoch_loss
        loss_list.append(epoch_loss)
        print('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end-start))
