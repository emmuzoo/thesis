





docker run -p 5000:5000 \
    -e MODEL_URI="models:/sentiment-analysis/1" \
    -e S3_BUCKET="my-s3-bucket" \
    -e VECTORIZER_URI="path/to/tfidf_vectorizer.pkl" \
    sentiment-classifiers


curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "I love this product!"}'
