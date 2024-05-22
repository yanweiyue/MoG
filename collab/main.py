from args import parser_loader
from datasets import get_dataset
import torch
from train_eval import cross_validation_with_val_set
from MoG import MoG

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


def main():
    args = parser_loader()
    device = torch.device("cuda:" + args['device']) if torch.cuda.is_available() else torch.device("cpu")

    dataset_name = 'COLLAB'
    results = []
    
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - MoG')
        
    dataset = get_dataset(dataset_name, sparse=True)
    print(dataset.data)

    loss, acc, std = cross_validation_with_val_set(
                dataset,
                folds=3,
                epochs=args['epochs'],
                batch_size=args['batch_size'],
                lr=args['lr'],
                lr_decay_factor=0.5,
                lr_decay_step_size=50,
                weight_decay=0,
                logger=None,
                args=args,
                device= device)
    if loss < best_result[0]:
        best_result = (loss, acc, std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
    results = '\n'.join(results)
    print(f'--\n{results}')

if __name__ == '__main__':
    main()