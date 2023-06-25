def train_model(model, n_epochs, optimizer, criterion, target, metric):

    with open('/content/gdrive/MyDrive/CNN_one_model/arousal_history4.csv', 'w') as file:
          writer = csv.writer(file)
          writer.writerows([['epoch', 'train_loss', 'train_metric', 'valid_loss', 'valid_metric']])
    model.to(device)

    history = {
    'train_losses': [],
    'valid_losses': [],
    'train_metrics': [],
    'valid_metrics': []
    }

    best_metrics = {'epoch': 0, 'valid_loss': 0, 'valid_metric': 0}

    for epoch in range(n_epochs):
        train_losses_iter = []
        train_metrics_iter = []
        model.train()
        j=0
        for music, arousal, valence in train_data:
            if j % 10 == 0:
              print(f'{j} итерация в train')
            j+=1
            music, arousal, valence = music.to(device), arousal.to(device), valence.to(device)
            out = model(music)
            if target == 'arousal':
                loss = torch.sqrt(criterion(out.squeeze(), arousal))
                metric_res = metric(out.squeeze(), arousal)
            elif target == 'valence':
                loss = torch.sqrt(criterion(out.squeeze(), valence))
                metric_res = metric(out.squeeze(), valence)

            train_losses_iter.append(loss.item())
            train_metrics_iter.append(metric_res.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history['train_losses'].append(np.mean(train_losses_iter))
        history['train_metrics'].append(np.mean(train_metrics_iter))

        valid_losses_iter = []
        valid_metrics_iter = []
        model.eval()

        i=0
        for music, arousal, valence in valid_data:
            i+= 1
            if i % 10 == 0:
                print(f'{i} итерация в valid')

            music, arousal, valence = music.to(device), arousal.to(device), valence.to(device)
            out = model(music)
            if target == 'arousal':
                loss = torch.sqrt(criterion(out.squeeze(), arousal))
                metric_res = metric(out.squeeze(), arousal)
            elif target == 'valence':
                loss = torch.sqrt(criterion(out.squeeze(), valence))
                metric_res = metric(out.squeeze(), valence)

            valid_losses_iter.append(loss.item())
            valid_metrics_iter.append(metric_res.item())

        history['valid_losses'].append(np.mean(valid_losses_iter))
        history['valid_metrics'].append(np.mean(valid_metrics_iter))
        with open('/content/gdrive/MyDrive/CNN_one_model/arousal_history4.csv', 'a') as file:
          writer = csv.writer(file)
          writer.writerows([[epoch, round(history["train_losses"][-1], 4),
                             round(history["train_metrics"][-1], 4),
                             round(history["valid_losses"][-1], 4),
                             round(history["valid_metrics"][-1], 4)]])

        if (best_metrics['valid_loss'] == 0) or (best_metrics['valid_loss'] > history["valid_losses"][-1]):
          best_metrics['epoch'] = epoch
          best_metrics['valid_loss'] = history["valid_losses"][-1]
          best_metrics['valid_metric'] = history["valid_metrics"][-1]
          torch.save(model.state_dict(), '/content/gdrive/MyDrive/CNN_one_model/arousal_weights4/best_model.pt')
          print(f"best results: epoch {best_metrics['epoch']}, valid loss {best_metrics['valid_loss']}, valid metric {best_metrics['valid_metric']}")

          with open('/content/gdrive/MyDrive/CNN_one_model/arousal_best_res4.json', 'w') as outfile:
            json.dump(best_metrics, outfile)

        torch.save(model.state_dict(), '/content/gdrive/MyDrive/CNN_one_model/arousal_weights4/each_epochs.pt')
        if epoch == 100:
          torch.save(model.state_dict(), '/content/gdrive/MyDrive/CNN_one_model/arousal_weights4/100_epochs.pt')
        if epoch == 199:
          torch.save(model.state_dict(), '/content/gdrive/MyDrive/CNN_one_model/arousal_weights4/200_epochs.pt')

        print(f'epoch: {epoch}\n'
        f'train: loss {history["train_losses"][-1]:.4f}\n'
        f'train: metric {history["train_metrics"][-1]:.4f}\n'
        f'valid: loss {history["valid_losses"][-1]:.4f}\n'
        f'valid: metric {history["valid_metrics"][-1]:.4f}')
        print(f'{"-"*35}')
        print()
    return history