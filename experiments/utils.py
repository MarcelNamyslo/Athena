import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import time
import os
import numpy as np


def save_results_to_pdf_experiment1(i, percentages, stats_lrp, stats_no_lrp, epochs, base_learning_rate, num_repeats, model_type, factor=1, epoch_wise_lrp=None, epoch_wise_no_lrp=None, time_no_lrp = None, time_lrp= None, pdf_filename='experiment_results.pdf'):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  

    model_type_name = model_type.__name__ if isinstance(model_type, type) else str(model_type)
    
    #
    output_dir = 'experiment 1 results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full filename path
    pdf_filename = os.path.join(output_dir, f'Experiment_Results_accuracy_comparison_{model_type_name}_{timestamp}_factor_{factor}_numRepeats_{num_repeats}_epochs_{epochs}_experiment7 mit last layer = false.pdf')

    
    with PdfPages(pdf_filename) as pdf:
        fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={'height_ratios': [1, 2]})  
        axs[0].text(0.1, 0.9, 'Experimental Setup', fontsize=12, weight='bold')  
        axs[0].text(0.1, 0.75, f'Epochs: {epochs}', fontsize=10)
        axs[0].text(0.1, 0.65, f'Base Learning Rate: {base_learning_rate}', fontsize=10)
        axs[0].text(0.1, 0.55, f'Number of Repeats: {num_repeats}', fontsize=10)
        axs[0].text(0.1, 0.45, f'Percentages: {percentages}', fontsize=10)
        axs[0].text(0.1, 0.35, f'Factor: {factor}', fontsize=10)  
        axs[0].axis('off')  
        axs[1].axis('off')  
        pdf.savefig(fig)
        plt.close(fig)

        for epoch_index in [-2, -1]:  
            epoch_number = epochs + epoch_index + 1  

            fig = plt.figure(figsize=(8.27, 11.69))  
            gs = GridSpec(2, 1, height_ratios=[1.5, 1]) 
            ax1 = fig.add_subplot(gs[0])
            ax2 = ax1.twinx()  
            reversed_percentages = percentages
            lrp_acc = [epoch[epoch_index] for epoch in epoch_wise_lrp] 
            no_lrp_acc = [epoch[epoch_index] for epoch in epoch_wise_no_lrp]  

            min_accuracy = min(min(lrp_acc), min(no_lrp_acc))
            max_accuracy = max(max(lrp_acc), max(no_lrp_acc))

            # Plot LRP accuracies
            color = 'tab:blue'
            ax1.set_xlabel(f'Percentage of Training Data', fontsize=9)  # Label x-axis
            ax1.set_ylabel('Accuracy with LRP', color=color, fontsize=9)
            ax1.plot(reversed_percentages, lrp_acc, color=color, marker='o', markersize=3, label=f'LRP (Epoch {epoch_number})')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(min_accuracy - 5, max_accuracy + 5)
            ax1.set_xlim(max(percentages), min(percentages)) 

            # Plot No LRP accuracies
            color = 'tab:red'
            ax2.set_ylabel('Accuracy without LRP', color=color, fontsize=9)
            ax2.plot(reversed_percentages, no_lrp_acc, color=color, marker='s', markersize=3, linestyle='--', label=f'No LRP (Epoch {epoch_number})')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(min_accuracy - 5, max_accuracy + 5)

            ax1.legend(loc='upper left', fontsize=8)
            ax2.legend(loc='upper right', fontsize=8)

            ax_table = fig.add_subplot(gs[1])
            ax_table.axis('off')  

            data = {
                'Percentage of Training Data': reversed_percentages,
                'Accuracy with LRP (%)': [f'{acc:.2f}' for acc in lrp_acc],
                'Accuracy without LRP (%)': [f'{acc:.2f}' for acc in no_lrp_acc]
            }
            df = pd.DataFrame(data)

            table = ax_table.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8) 
            table.scale(0.8, 0.8)  

            fig.tight_layout()
            pdf.savefig(fig)  
            plt.close(fig)

        data = {
            'Percentage of Training Data (%)': percentages,
            'Training Time with LRP (s)': [f'{t:.2f}' for t in time_lrp],
            'Training Time without LRP (s)': [f'{t:.2f}' for t in time_no_lrp],
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved as {pdf_filename}")
    
    
def save_results_to_pdf_experiment2(i, percentages, stats_lrp, stats_no_lrp, epochs, base_learning_rate, num_repeats, model_type, apply_lrp_to = "all", factor=10, time_lrp = None, time_no_lrp= None, pdf_filename='experiment_results.pdf'):

    print("stats_lrp")
    print(stats_lrp)
    print("stats_no_lrp")
    print(stats_no_lrp)
    timestamp = time.strftime("%Y%m%d-%H%M%S") 

    model_type_name = model_type.__name__ if isinstance(model_type, type) else str(model_type)

    output_dir = 'experiment 2 results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stats_lrp = [0 if np.isinf(x) else x for x in stats_lrp]
    stats_no_lrp = [0 if np.isinf(x) else x for x in stats_no_lrp]


    pdf_filename = os.path.join(output_dir, f'Experiment_Results_accuracy_comparison_{model_type_name}_{timestamp}_factor_{factor}_numRepeats_{num_repeats}_applied_to_{apply_lrp_to}].pdf')


    with PdfPages(pdf_filename) as pdf:
    
        fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={'height_ratios': [1, 2]})  

        axs[0].text(0.1, 0.9, 'Experimental Setup', fontsize=12, weight='bold') 
        axs[0].text(0.1, 0.75, f'Epochs: {epochs}', fontsize=10)
        axs[0].text(0.1, 0.65, f'Base Learning Rate: {base_learning_rate}', fontsize=10)
        axs[0].text(0.1, 0.55, f'Number of Repeats: {num_repeats}', fontsize=10)
        axs[0].text(0.1, 0.45, f'Percentages: {percentages}', fontsize=10)
        axs[0].text(0.1, 0.35, f'Factor: {factor}', fontsize=10) 
        axs[0].text(0.1, 0.25, f'Linear or all: {apply_lrp_to}', fontsize=10) 
        axs[0].axis('off')  

        ax1 = axs[1]  
        ax2 = ax1.twinx() 

        min_accuracy = min(min(stats_lrp), min(stats_no_lrp))
        max_accuracy = max(max(stats_lrp), max(stats_no_lrp))
        
        color = 'tab:blue'
        ax1.set_xlabel('Percentage of Training Data', fontsize=10) 
        ax1.set_ylabel('Accuracy with LRP', color=color, fontsize=10)
        ax1.plot(percentages, stats_lrp, color=color, marker='o', markersize=3, label='With LRP')  
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(min_accuracy - 5, max_accuracy + 5)
        
        color = 'tab:red'
        ax2.set_ylabel('Accuracy without LRP', color=color, fontsize=10)
        ax2.plot(percentages, stats_no_lrp, color=color, marker='s', markersize=3, linestyle='--', label='Without LRP')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(min_accuracy - 5, max_accuracy + 5)

        ax1.set_xlim(max(percentages), min(percentages))  
        fig.tight_layout() 

        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))  
        ax.axis('off')  
        data = {
            'Percentage of Training Data (%)': percentages,
            'Training Time with LRP (s)': [f'{t:.2f}' for t in time_lrp],
            'Training Time without LRP (s)': [f'{t:.2f}' for t in time_no_lrp],
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved as {pdf_filename}")