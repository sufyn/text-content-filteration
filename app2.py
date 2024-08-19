import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the model and vectorizer 
loaded_model = joblib.load('logistic_regression_model.pkl')
vect = joblib.load('vectorizer.pkl')

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

st.title("Toxicity Prediction App")

user_input = st.text_area("Enter text for toxicity prediction:")

if st.button("Predict"):
    if user_input:
        # Clean 
        cleaned_text = clean_text(user_input)

        # Transform 
        new_text_vec = vect.transform([cleaned_text])

        # Predict 
        prediction = loaded_model.predict_proba(new_text_vec)[:, 1][0]
        val = prediction * 10000
        
        st.write(f"Abusiveness Percentage: {prediction * 10000:.2f}%")

        
        if val > 50:
            st.error("This text is classified as abusive.")
        else:
            st.success("This text is not classified as abusive.")
    else:
        st.warning("Please enter some text.")                                                                                             [combine below and above code]                                                                                                         import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create the DataFrame with categories and keywords
data = pd.DataFrame({
    'Category': [
        'Travel & Transport', 'Education', 'Worker', 'Food', 'Real Estate',
        'Store', 'Consultant', 'Freelancer', 'Donate', 'Stall',
        'Repair', 'Daily Services', 'Horoscope', 'Lifestyle', 'Arts',
        'Rent/Lease', 'IT & Hardware', 'Fitness', 'Resell', 'Beauty',
        'Health', 'Fashion', 'Medical Care'
    ],
    'Keywords': [
        'trip vacation flight bus train taxi cab tourism hotel booking tour guide travel package road trip travel insurance',
        'school college university course tuition learning exam teacher student scholarship online course certification study material coaching curriculum academic e-learning',
        'labor job employment hire workforce construction worker handyman skilled labor temporary work contract work blue-collar factory worker labor market',
        'restaurant dining cuisine meal recipe cooking chef fast food takeout catering food delivery grocery ingredients organic food bakery café culinary',
        'property house apartment rent lease buy sell realtor mortgage realty commercial property land housing market home residential investment property',
        'shop retail shopping boutique outlet e-commerce marketplace sale discount product merchandise inventory store location storefront mall online store',
        'advisory consulting strategy business consultant management financial advisor IT consultant legal advice market analysis consultancy expert advice project management consulting services',
        'freelance self-employed contract work gig economy remote work independent contractor freelance job freelancing platform portfolio client freelance projects upwork fiverr freelancing services',
        'charity donation fundraiser philanthropy give support non-profit donate money volunteer donate clothes humanitarian crowdfunding relief fund donation drive',
        'market vendor booth kiosk fair exhibition pop-up shop trade show food stall street vendor stall holder temporary shop',
        'fix maintenance service mechanic electrician technician repair shop home repair auto repair appliance repair electronic repair fix it mending restoration',
        'housekeeping cleaning laundry delivery service errands pet care babysitting daily help domestic service personal care routine service grocery delivery',
        'astrology zodiac star sign horoscope reading tarot daily horoscope astrological sign horoscope prediction birth chart astrology report compatibility',
        'fashion trends wellness luxury lifestyle tips home decor lifestyle blog healthy living lifestyle brand influencer social life lifestyle habits lifestyle choices modern living sports fitness health active',
        'painting sculpture gallery exhibition artwork artist creative photography performing arts fine arts art class art supplies design crafts art museum',
        'rental leasing property lease equipment rental car rental apartment lease office space rent agreement short-term lease rental service lease contract',
        'computer software hardware IT support network server computer repair IT services tech support gadgets technology IT solutions hardware upgrade system installation',
        'sport gym exercise workout health personal trainer yoga fitness class fitness equipment physical fitness diet weight loss fitness plan running bodybuilding aerobics sports athletics training',
        'resale secondhand used items resale market thrift consignment pre-owned resale value flipping buy and sell reselling platform secondhand shop resale business',
        'makeup skincare beauty products salon cosmetics hair care beauty tips spa manicure pedicure beauty routine beauty treatment beauty salon beauty trends',
        'healthcare medicine doctor hospital wellness health tips mental health fitness medical treatment health insurance pharmacy healthy lifestyle health clinic health checkup disease prevention',
        'clothing style fashion trends designer fashion show apparel boutique fashion accessories wardrobe runway fashion design fashion brand shopping fashion blog outfit',
        'healthcare doctor hospital clinic medical treatment surgery healthcare services medical checkup health care provider nurse patient care medical consultation healthcare professional healthcare facility'
    ]
})

# Preprocess the keywords by splitting them into individual words
data['Keywords'] = data['Keywords'].apply(lambda x: ' '.join(x.split()))

# Vectorize the keywords using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Keywords'])

# Convert the TF-IDF matrix to an array for cosine similarity calculations
tfidf_array = tfidf_matrix.toarray()

# Function to get category suggestions based on user input
def get_category_suggestions(user_input, tfidf_vectorizer, tfidf_array, data):
    # Transform the user input into the same TF-IDF space
    user_input_tfidf = tfidf_vectorizer.transform([user_input]).toarray()

    # Calculate cosine similarity between user input and each category's keywords
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_array).flatten()

    # Get the indices of the top N categories with highest similarity
    top_indices = cosine_similarities.argsort()[-3:][::-1]

    # Retrieve the corresponding categories
    suggestions = [data['Category'][i] for i in top_indices]

    return suggestions

user_input = "sports"
suggestions = get_category_suggestions(user_input, tfidf_vectorizer, tfidf_array, data)
print(f"Suggestions for '{user_input}': {suggestions}")
