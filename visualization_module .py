import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def read_metrics(filename):
    """Read metrics from a given text file."""
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            if ': ' in line:
                key, value = line.split(': ')
                data[key.strip()] = value.strip()
    return data

def collect_all_metrics(filenames):
    """Collect metrics from multiple files into a DataFrame."""
    data = []
    for filename in filenames:
        metrics = read_metrics(filename)
        metrics['Algorithm'] = filename.replace('.txt', '')  # Use file name as algorithm name
        data.append(metrics)
    return pd.DataFrame(data)

def convert_metrics_to_numeric(df):
    """Convert metric columns to numeric types."""
    for col in ['Execution Time', 'Best Fitness', 'Coverage']:
        if col in df:
            df[col] = pd.to_numeric(df[col].str.replace('%', ''), errors='coerce')
    return df


if __name__ == '__main__':
    filenames = [
        'new_hybrid_algorithm_metrics.txt',
        'genetic_metrics.txt',
        'simulated_anne_metrics.txt',
        'cuckoo_metrics.txt',
        'TLBOalgorithm_metrics.txt'
    ]

    # Collect all metrics into a DataFrame
    df = collect_all_metrics(filenames)
    df = convert_metrics_to_numeric(df)

    # Reshape the DataFrame for Execution Time comparison
    df_exec_time = df[['Algorithm', 'Execution Time']].melt(id_vars='Algorithm', var_name='Metric', value_name='Value')

    # Plot Execution Time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Value', data=df_exec_time)
    plt.title('Execution Time Comparison Across Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Reshape the DataFrame for Best Fitness and Coverage comparison
    df_fitness_coverage = df[['Algorithm', 'Best Fitness', 'Coverage']].melt(id_vars='Algorithm', var_name='Metric',
                                                                             value_name='Value')

    # Plot Best Fitness and Coverage comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Value', hue='Metric', data=df_fitness_coverage)
    plt.title('Best Fitness and Coverage Comparison Across Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()