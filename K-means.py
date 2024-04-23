import pandas as pd
import numpy as np
from tkinter import *
from tkinter import messagebox, filedialog

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    file_entry.delete(0, END)
    file_entry.insert(END, file_path)

def get_labels(distances):
    labels = np.zeros(distances.shape[0], dtype=np.int64) # #rows only
    for i in range(len(distances)):
        min_distance = float('inf')
        idx = 0
        for j in range(len(distances[i])):
            if distances[i][j] < min_distance:
                min_distance = distances[i][j]
                idx = j
        labels[i] = idx
    return labels

def generate_output_command():
    try:
        file_path = file_entry.get()
        percentage = int(percentage_entry.get())
        k = int(k_entry.get())

        # Load the dataset
        df = pd.read_csv(file_path)

        num_records = int(len(df) * (percentage / 100))
        df = df.head(num_records)

        X = df[['IMDB Rating']]

        z_scores = (X - X.mean()) / X.std() 
        
        # df = df.values

        # filtered_df, xa = iqr(df)
        # filtered_X = filtered_df[['IMDB Rating']]
        # Filter out outliers based on Z-score mean
        threshold = 2
        filtered_df = df[(z_scores.abs() < threshold).all(axis=1)] ## Add iff z_scores < threshold (for .all elements)

        filtered_X = filtered_df[['IMDB Rating']]

        #np.random.seed(42)
        centroids_indices = np.random.choice(filtered_X.index, size=k, replace=False)
        centroids = filtered_X.loc[centroids_indices].values.flatten()

        max_iterations = 100
        for _ in range(max_iterations):
            distances = np.zeros((len(filtered_X), k))
            for i, data_point in enumerate(filtered_X.values): # eloop over filtered_X without outliers
                distances[i] = np.sqrt((data_point - centroids) ** 2) # for each centroid
            labels = get_labels(distances)
            new_centroids = np.concatenate([filtered_X[labels == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids.flatten()):
                break
            
            centroids = new_centroids.flatten()

        # Group movies into clusters
        filtered_df['Cluster'] = labels
        print(filtered_df)

        # Detect and output outlier records
        output = ""
        outliers = df[~df.index.isin(filtered_df.index)] #indcies 0 1 2 3 ... 2 3 => result 0 1
        if not outliers.empty:
            cnt = 0
            output += "Outlier Movies:\n"
            for index, row in outliers.iterrows():
                cnt += 1
                output += f"{row['Movie Name']} (IMDB Rating: {row['IMDB Rating']})\n"
            print(f"#Outliers: {cnt}")
            output += "##################################\n"

        # Output k lists of movies grouped by clusters
        for cluster_id in range(k):
            output += f"Cluster {cluster_id + 1}:\n"
            movies_in_cluster = filtered_df[filtered_df['Cluster'] == cluster_id][['Movie Name', 'IMDB Rating']].values
            cnt= 0
            for movie, rating in movies_in_cluster:
                cnt += 1
                output += f"{movie} (IMDB Rating: {rating})\n"

            print(f"#Movies in Cluster {cluster_id + 1}: {cnt}")
            output += "##################################\n"

        # Show the output
        output_text.delete('1.0', END)
        output_text.insert(END, output)

    except Exception as e:
        messagebox.showerror("Error", str(e))

window = Tk()
window.title("Clustering and Outlier Detection")
window.configure(background="#f0f0f0")

file_label = Label(window, text="Select CSV File:")
file_label.grid(row=0, column=0, padx=5, pady=5)
file_entry = Entry(window)
file_entry.grid(row=0, column=1, padx=5, pady=5)
browse_button = Button(window, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=5, pady=5)

percentage_label = Label(window, text="Percentage of Data (%):")
percentage_label.grid(row=1, column=0, padx=5, pady=5)
percentage_entry = Entry(window)
percentage_entry.grid(row=1, column=1, padx=5, pady=5)

k_label = Label(window, text="Number of Clusters (K):")
k_label.grid(row=2, column=0, padx=5, pady=5)
k_entry = Entry(window)
k_entry.grid(row=2, column=1, padx=5, pady=5)

generate_button = Button(window, text="Generate Output", command=generate_output_command)
generate_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

output_text = Text(window, height=20, width=50, bg="#ffffff", fg="blue")
output_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

scrollbar = Scrollbar(window, command=output_text.yview)
scrollbar.grid(row=4, column=3, sticky='ns')
output_text.config(yscrollcommand=scrollbar.set)

window.grid_rowconfigure(4, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=1)
window.grid_columnconfigure(3, weight=0)  # Adjusted column weight for the scrollbar

window.mainloop()
