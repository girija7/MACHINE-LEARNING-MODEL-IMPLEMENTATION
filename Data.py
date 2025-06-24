 # Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only label & text columns
df.columns = ['label', 'text']

# Convert labels to binary (0=ham, 1=spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check class distribution
print(df['label'].value_counts())
sns.countplot(data=df, x='label')
plt.title("Spam vs Ham Distribution")
plt.show()
