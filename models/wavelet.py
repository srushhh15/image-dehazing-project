import torch
import pywt


def dwt(x):
    """
    Discrete Wavelet Transform (Haar)
    """

    b, c, h, w = x.shape

    LL = []
    LH = []
    HL = []
    HH = []

    for i in range(b):

        img = x[i].cpu().numpy()

        ll_channels = []
        lh_channels = []
        hl_channels = []
        hh_channels = []

        for ch in range(c):

            coeffs = pywt.dwt2(img[ch], 'haar')

            ll, (lh, hl, hh) = coeffs

            ll_channels.append(ll)
            lh_channels.append(lh)
            hl_channels.append(hl)
            hh_channels.append(hh)

        LL.append(ll_channels)
        LH.append(lh_channels)
        HL.append(hl_channels)
        HH.append(hh_channels)

    LL = torch.tensor(LL).to(x.device)
    LH = torch.tensor(LH).to(x.device)
    HL = torch.tensor(HL).to(x.device)
    HH = torch.tensor(HH).to(x.device)

    return LL, LH, HL, HH