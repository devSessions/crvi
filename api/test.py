import predict


url='http://res.cloudinary.com/imbiswas/image/upload/v1514040157/20_wvtwwg.jpg'
sender=predict.predict(url)
rec=sender.predict_only()
print(rec)
print(type(rec))
