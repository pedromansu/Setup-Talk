from PIL import Image
import matplotlib.pyplot as plt

idx_to_class = ()

def predict(model, test_image_name):
    idx_to_class = {0: 'bear', 1: 'giraffe', 2: 'gorilla', 3: 'llama', 4: 'zebra'}
    
    
    transform = image_transforms['test']
 
    test_image = Image.open(test_image_name)
    plt.imshow(test_image)
     
    test_image_tensor = transform(test_image)
 
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
     
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(5, dim=1)
        print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])