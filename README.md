<h1>Sentiment Analysis PCA</h1>

<p>This project performs sentiment analysis on customer feedback data using various natural language processing (NLP) techniques and clustering methods. The script processes the text data, identifies keywords and phrases, and visualizes the results using word clouds and scatter plots.</p>

<h2>Directory Structure</h2>
<pre>
your_project_directory/
│
├── culture_sentiment_analysis_pca/
│   ├── sentiment_analysis_pca.py
│   ├── data/
│   │   ├── comments.csv
│   │   └── verbatim.csv
│   └── outputs/
</pre>

<h2>Setup</h2>
<ol>
    <li>Create a virtual environment and activate it:
        <pre><code>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>
    </li>
    <li>Install the required packages:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Create the necessary directories:
        <pre><code>mkdir -p culture_sentiment_analysis_pca/data
mkdir -p culture_sentiment_analysis_pca/outputs</code></pre>
    </li>
    <li>Place your input CSV files (`comments.csv` and `verbatim.csv`) in the `data` directory.</li>
</ol>

<h2>Running the Script</h2>
<ol>
    <li>Navigate to the project directory:
        <pre><code>cd culture_sentiment_analysis_pca</code></pre>
    </li>
    <li>Run the script:
        <pre><code>python sentiment_analysis_pca.py</code></pre>
    </li>
</ol>

<h2>Script Overview</h2>
<p>The script performs the following steps:</p>
<ol>
    <li>Loads the input data from CSV files.</li>
    <li>Combines the datasets into a single DataFrame.</li>
    <li>Processes the text data by applying stemming and lemmatization.</li>
    <li>Filters the data to keep only rows containing specified keywords or phrases.</li>
    <li>Counts the occurrences of each keyword and phrase.</li>
    <li>Visualizes the keyword and phrase counts using bar charts.</li>
    <li>Generates a word cloud for the filtered text data.</li>
    <li>Applies K-means clustering to the text data.</li>
    <li>Reduces the dimensionality of the data using PCA for visualization purposes.</li>
    <li>Saves the filtered data with clusters to a new CSV file.</li>
    <li>Prints representative texts for each cluster.</li>
    <li>Adds a new column with related keywords or phrases and saves the final data to a new CSV file.</li>
</ol>

<h2>Output</h2>
<p>The script generates the following output files in the <code>outputs</code> directory:</p>
<ul>
    <li><code>filtered_combined_data.csv</code>: The filtered data with clusters.</li>
    <li><code>final_output.csv</code>: The final data with related keywords or phrases.</li>
</ul>

<h2>Dependencies</h2>
<p>The script requires the following Python packages:</p>
<ul>
    <li>pandas</li>
    <li>collections</li>
    <li>matplotlib</li>
    <li>wordcloud</li>
    <li>nltk</li>
    <li>fuzzywuzzy</li>
    <li>textblob</li>
    <li>scikit-learn</li>
    <li>spacy</li>
    <li>numpy</li>
</ul>

<p>Ensure you have these packages installed before running the script.</p>

</body>
</html>