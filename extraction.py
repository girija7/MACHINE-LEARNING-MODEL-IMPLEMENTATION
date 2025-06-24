tfidf = TfidfVectorizer(max_features=5000)  # Top 5000 words
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# Split into train-test (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
