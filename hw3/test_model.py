import torch
import torch.nn as nn
import random

from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE
from model import *


#######################################
###             Testing             ###
#######################################


def allclose(actual, expected):
    if isinstance(expected, list):
        return any(allclose(actual, x) for x in expected)
    return actual.allclose(expected, atol=1e-5)


def assert_allclose(actual, expected, tag='Frame loss'):
    assert allclose(actual, expected), f'\n{tag} does not match expected result.'


def test_forward_and_backward(data, model_class, expected=None):
    audio = data['audio'].unsqueeze(0)
    frame = data['frame'].unsqueeze(0)
    onset = data['onset'].unsqueeze(0)

    cnn_unit, fc_unit = 48, 256
    model = model_class(cnn_unit=cnn_unit, fc_unit=fc_unit)
    criterion = nn.BCEWithLogitsLoss()

    print('Testing', model_class.__name__, 'forward', end='...')
    frame_logit, onset_logit = model(audio)
    loss_frame = criterion(frame_logit, frame)
    loss_onset = criterion(onset_logit, onset)

    print(loss_frame.detach())
    print(expected[0])
    assert_allclose(loss_frame.detach(), expected[0], "Frame loss")
    assert_allclose(loss_onset.detach(), expected[1], "Onset loss")
    assert_allclose(frame_logit[0, :3, :3], expected[2], "Frame logit")
    assert_allclose(onset_logit[0, :3, :3], expected[3], "Onset logit")
    print('passed!')

    print('Testing', model_class.__name__, 'backward', end='...')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    (loss_frame + loss_onset).backward()
    optimizer.step()

    frame_logit, onset_logit = model(audio)
    loss_frame = criterion(frame_logit, frame)
    loss_onset = criterion(onset_logit, onset)

    assert_allclose(loss_frame.detach(), expected[4], "Frame loss")
    assert_allclose(loss_onset.detach(), expected[5], "Onset loss")
    assert_allclose(frame_logit[0, :3, :3], expected[6], "Frame logit")
    assert_allclose(onset_logit[0, :3, :3], expected[7], "Onset logit")
    print('passed!')


def test_question_1(data):
    expected = [
        torch.tensor(0.69820082),
        torch.tensor(0.68893498),
        torch.tensor([[-0.07700811, 0.01711646, 0.07250410],
                      [-0.05614455, 0.04127759, 0.06543481],
                      [-0.02837110, 0.03935390, 0.06058343]]),
        torch.tensor([[-0.03639920, -0.05293848, -0.08305336],
                      [-0.00593201, -0.04121016, -0.10712688],
                      [-0.00755560, -0.04383054, -0.13804567]]),
        torch.tensor(0.69819885),
        torch.tensor(0.68893188),
        torch.tensor([[-0.07701070, 0.01711437, 0.07250243],
                      [-0.05614709, 0.04127425, 0.06543226],
                      [-0.02837374, 0.03934879, 0.06058135]]),
        torch.tensor([[-0.03641617, -0.05296630, -0.08305261],
                      [-0.00594394, -0.04123839, -0.10713291],
                      [-0.00756159, -0.04385110, -0.13805026]])
    ]
    test_forward_and_backward(data, Transcriber_RNN, expected)


def test_question_2(data):
    expected = [
        torch.tensor(0.69386131),
        torch.tensor(0.69428575),
        torch.tensor([[-0.01180422, -0.09453917, -0.00734041],
                      [-0.00751195, -0.11016708, -0.01735646],
                      [-0.05415545, -0.10199113, 0.00362945]]),
        torch.tensor([[0.00959453, -0.08459471, -0.01636595],
                      [0.01384565, -0.06305125, -0.03657395],
                      [-0.02366585, -0.03840892, -0.01877560]]),
        torch.tensor(0.69492340),
        torch.tensor(0.69348210),
        torch.tensor([[0.01592010, -0.08493100, 0.03172420],
                      [-0.02577951, -0.09247213, 0.04698481],
                      [-0.03992629, -0.08237503, 0.05112319]]),
        torch.tensor([[-0.00354448, -0.09177583, -0.02958333],
                      [-0.01413403, -0.08793411, -0.01763032],
                      [-0.00520566, -0.04321136, -0.01629483]]),
    ]
    test_forward_and_backward(data, Transcriber_CRNN, expected)


def test_question_3(data):
    expected = [
        [torch.tensor(0.69590342),
         torch.tensor(0.69795382)],
        torch.tensor(0.69238877),
        [
            torch.tensor([[0.06370672, 0.08482580, 0.06356754],
                          [0.06739222, 0.07954120, 0.07194521],
                          [0.06046356, 0.09059811, 0.07633506]]),
            torch.tensor([[0.05622103, 0.05425581, 0.03362592],
                          [0.05545133, 0.05858815, 0.05857687],
                          [0.05987605, 0.05469957, 0.07613088]])
        ],
        torch.tensor([[0.06340785, -0.06165579, 0.01130402],
                      [0.03822849, -0.06998032, -0.02159066],
                      [0.04425360, -0.05165110, -0.02422101]]),
        [torch.tensor(0.69502985),
         torch.tensor(0.69740343)],
        torch.tensor(0.69107497),
        [
            torch.tensor([[0.05181099, 0.07332264, 0.06701970],
                          [0.04980478, 0.07966238, 0.07465430],
                          [0.05082099, 0.08482911, 0.07430131]]),
            torch.tensor([[0.05188487, 0.05187126, 0.02359203],
                          [0.05006664, 0.06773943, 0.04167914],
                          [0.06316102, 0.08043950, 0.05361710]])
        ],
        torch.tensor([[0.12146214, -0.10427066, -0.04547792],
                      [0.12515414, -0.12060987, -0.04273968],
                      [0.08767528, -0.12075926, -0.07124287]]),
    ]
    test_forward_and_backward(data, Transcriber_ONF, expected)


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    from dataset import MAESTRO_small
    dataset = MAESTRO_small(groups=['debug'],
                            sequence_length=16000,
                            hop_size=HOP_SIZE,
                            random_sample=True)
    data = dataset[0]
    # test_question_1(data)       # Onset loss
    # test_question_2(data)       # Frame loss
    test_question_3(data)       # Frame loss
    print('All tests passed!')
