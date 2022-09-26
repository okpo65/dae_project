# test meta model with DAE
import torch
import numpy as np
import pandas as pd
def test_mlp(dae,
             mlp,
             test_dl):
    predictions = []
    dae.eval()
    mlp.eval()
    with torch.no_grad():
        for _, x in enumerate(test_dl):
            x = dae.feature(x.cuda())
            prediction = mlp.forward(x)
            predictions.append(prediction.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    # predictions = np.array(list(predictions.reshape(1, -1)[0]))#pd.DataFrame({'prediction': list(predictions.reshape(1, -1)[0])})
    return predictions

def test_tree_model(model_list,
                    dae,
                    test_dl):

    with torch.no_grad():
        dae_x = dae.feature(torch.Tensor(test_dl.dataset.x).to(torch.device('cuda')))
        dae_x = dae_x.detach().cpu().numpy()

    x_test = pd.DataFrame(dae_x, columns=[idx for idx in range(dae_x.shape[1])])
    raw_predictions = [model.eval_model(x_test)[:, 1] for model in model_list]
    predictions = sum(raw_predictions) / len(model_list)
    # predictions = pd.DataFrame(predictions, columns=['prediction'])
    return predictions, raw_predictions
