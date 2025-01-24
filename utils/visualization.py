import matplotlib.pyplot as plt
import numpy as np

def plot_class_accuracies(pre_acc, post_acc, classes_to_unlearn, logger):
    plt.figure(figsize=(15, 8))
    x = np.arange(len(pre_acc))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, pre_acc, width, label='Before Unlearning', 
                     color=['red' if i in classes_to_unlearn else 'steelblue' for i in range(len(pre_acc))])
    rects2 = plt.bar(x + width/2, post_acc, width, label='After Unlearning',
                     color=['darkred' if i in classes_to_unlearn else 'lightblue' for i in range(len(post_acc))])
    
    plt.xlabel('Class Index', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title('Class-wise Accuracy Before and After Unlearning', fontsize=20, pad=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
    
    autolabel(rects1)
    autolabel(rects2)
    
    for class_idx in classes_to_unlearn:
        plt.axvspan(class_idx-0.5, class_idx+0.5, color='yellow', alpha=0.2)
    
    plt.xticks(x, [f'Class {i}' for i in range(len(pre_acc))], rotation=45)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(logger.log_dir / 'class_accuracies.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curve(metrics, logger):
    rounds = [loss['round'] for loss in metrics['train_losses']]
    losses = [loss['loss'] for loss in metrics['train_losses']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, losses, 'b-', alpha=0.6)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(logger.log_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_trend(metrics, logger):
    rounds = [acc['round'] for acc in metrics['round_accuracies']]
    accuracies = [acc['accuracy'] for acc in metrics['round_accuracies']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, accuracies, 'g-', alpha=0.6)
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(logger.log_dir / 'accuracy_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
