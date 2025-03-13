def compare_models():
    print("Running manually tuned model...")
    manual_accuracy, manual_precision, manual_recall, manual_f1, manual_time, y_test_classes, y_pred_classes = manually_tuned_model()

    results['Manual Model'] = {
        'Accuracy': manual_accuracy,
        'Precision': manual_precision,
        'Recall': manual_recall,
        'F1-Score': manual_f1,
        'Training Time': manual_time,
    }

    print("\nRunning evolutionary algorithm...")
    evolutionary_algorithm_with_tracking(generations=3, population_size=5, num_parents=5)

    comparison_df = pd.DataFrame(results)
    print("\nComparison Table")
    print(comparison_df.to_string(index=False))

    comparison_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Performance')
    plt.legend()
    plt.show()

# Execute the comparison
compare_models()