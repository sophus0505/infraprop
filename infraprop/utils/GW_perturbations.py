import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline


def get_GW_newparam(DMAX=8000, DK_RANGE=50, FACT=0.5, NRAND=50, lambda_max_km=150, dlambda_km=1):
    lambda_b_m = 500
    m_b_m = 2 * np.pi / lambda_b_m
    alpha = 0.62
    N, f = 2 * np.pi / (5 * 60), 2 * np.pi / (18.7 * 60 * 60)
    p, q, s = 2, 3, 2
    SH_m = 7000
    dlambda_m = dlambda_km * 1000
    lambda_min_m = dlambda_m
    lambda_max_m = lambda_max_km * 1000 + lambda_min_m
    lambda_m = np.arange(lambda_min_m, lambda_max_m + dlambda_m, dlambda_m)
    dinv_lambda_m = 1 / (lambda_max_m - lambda_min_m)
    inv_lambda_m = np.arange(1 / lambda_max_m, 1 / lambda_min_m + dinv_lambda_m, dinv_lambda_m)
    m_z_m = 2 * np.pi * inv_lambda_m
    DH_km, DH_max_km = 10, 150
    H_m = np.arange(DH_km, DH_max_km + DH_km, DH_km) * 1000
    m_star_m = 2 * np.pi * np.exp(-(H_m) / (q + s) / SH_m) / 1000
    Fu_sat = 2 * np.pi * alpha * (N**2) / (m_z_m**3) / 3
    Fu_turb = 2 * np.pi * alpha * (N**2) * ((m_b_m / m_z_m) ** (5 / 3)) / (m_b_m**3) / 3
    V_new = np.zeros((len(H_m), NRAND, len(m_z_m) * 2 - 2))
    Fu_nosat = FACT * 2 * np.pi * alpha * (N**2) * ((m_z_m / m_star_m[:, None]) ** s) / (m_star_m[:, None] ** 3) / 3
    Fu_all = np.tile(FACT * Fu_sat, (len(H_m), 1))
    for k in range(len(H_m)):
        ind1 = np.where(m_z_m < m_star_m[k])[0]
        Fu_all[k, ind1] = Fu_nosat[k, ind1]
        ind2 = np.where(m_z_m > m_b_m)[0]
        Fu_all[k, ind2] = FACT * Fu_turb[ind2]
        absFU = np.sqrt(Fu_all[k, :])
        for n in range(NRAND):
            phase_rand = 2 * np.pi * np.random.rand(len(absFU))
            realFU = np.concatenate(
                [
                    absFU * np.cos(phase_rand),
                    absFU[-2:0:-1] * np.cos(phase_rand[-2:0:-1]),
                ]
            )
            imagFU = np.concatenate(
                [
                    absFU * np.sin(phase_rand),
                    -absFU[-2:0:-1] * np.sin(phase_rand[-2:0:-1]),
                ]
            )
            FU = realFU + 1j * imagFU
            FUI = FU - np.mean(FU)
            Z_PERT = np.arange(len(FUI)) / (1 / lambda_min_m - 1 / lambda_max_m) / 2
            fact_ifft = np.sqrt((lambda_max_m - lambda_min_m)) / len(FUI)
            V_new[k, n, :] = np.real(np.fft.ifft(FUI) / fact_ifft)
    ZT, DZ = 110, 20
    W = np.ones(Z_PERT.shape)
    indz = np.where(Z_PERT / 1000 > ZT)[0]
    ZW = np.exp(-(((Z_PERT / 1000 - ZT) / DZ) ** 2))
    W[indz[0] :] = ZW[indz[0] :]
    a = 1.3015
    sigma = np.sqrt(8 * np.log(a) * 2 * np.pi) / 2.108
    UVPERT_1D = np.zeros((NRAND, len(V_new[0, 0])))
    for n in range(NRAND):
        for k in range(len(H_m)):
            UVPERT_1D[n, :] += (
                W * np.squeeze(V_new[k, n]).T * (1 / sigma) * np.exp(-np.log(a) * ((Z_PERT / 1000 - H_m[k] / 1000) / DH_km * 2) ** 2)
            )
    h_rms_square = ((p - 1) * (s + 1) / (p + s)) * (1 / (3 - p) + np.log(m_b_m / m_star_m)) * (f / N) ** (p - 1) * m_star_m**2
    m_rms_square = 2 * ((s + 1) / (s + 3)) * (1 / (s + 3) + np.log(m_b_m / m_star_m)) * m_star_m**2
    lambda_corr_xy = np.sqrt(2) * 4 * np.pi / np.sqrt(h_rms_square)
    lambda_corr_z = 2 * np.pi / np.sqrt(m_rms_square)
    MMAX = 0
    for k in range(len(H_m)):
        MAX = np.arange(1, 2 * DMAX, 1 + round(lambda_corr_xy[k] / 1000))
        ICORR_MAX = MAX[-1] - 2
        if ICORR_MAX > MMAX:
            MMAX = ICORR_MAX
    DIST_2D0 = np.arange(0, MMAX + DK_RANGE, DK_RANGE)
    UPERT_CORR_2D0 = np.zeros((len(Z_PERT), len(DIST_2D0)))
    VPERT_CORR_2D0 = np.zeros((len(Z_PERT), len(DIST_2D0)))
    for k in range(len(H_m)):
        LCORR = round(lambda_corr_xy[k] / 1000)
        ICORR = np.arange(1, 2 * DMAX, 1 + round(lambda_corr_xy[k] / 1000))
        for m in range(len(ICORR) - 1):
            irandu = np.random.randint(1, NRAND)
            irandv = np.random.randint(1, NRAND)
            vertical_weight = W * np.exp(-np.log(1.5) * ((Z_PERT / 1000 - H_m[k] / 1000) / DH_km * 2) ** 2)
            horizontal_weight = np.exp(-np.log(1.5) * ((DIST_2D0 - ICORR[m + 1] + 1 + LCORR / 2) / LCORR * 2) ** 2)
            upert_corr_vert = np.tile(UVPERT_1D[irandu, :] * vertical_weight, (len(DIST_2D0), 1)).T
            vpert_corr_vert = np.tile(UVPERT_1D[irandv, :] * vertical_weight, (len(DIST_2D0), 1)).T
            UPERT_CORR_2D0 += upert_corr_vert * np.tile(horizontal_weight, (len(Z_PERT), 1))
            VPERT_CORR_2D0 += vpert_corr_vert * np.tile(horizontal_weight, (len(Z_PERT), 1))
    DIST_2D = np.arange(0, MMAX + 1, DK_RANGE)

    # Old interpolation method with deprecated interp2d
    # interp_func_upert = interp2d(DIST_2D0, Z_PERT, UPERT_CORR_2D0, kind="cubic")
    # interp_func_vpert = interp2d(DIST_2D0, Z_PERT, VPERT_CORR_2D0, kind="cubic")

    # UPERT_CORR_2D, VPERT_CORR_2D = (
    #     interp_func_upert(DIST_2D, Z_PERT),
    #     interp_func_vpert(DIST_2D, Z_PERT),
    # )

    # New interpolation with equivalent RectBivariateSpline
    DIST_2D = np.arange(0, MMAX + 1, DK_RANGE)

    spline_upert = RectBivariateSpline(Z_PERT, DIST_2D0, UPERT_CORR_2D0, kx=3, ky=3)
    spline_vpert = RectBivariateSpline(Z_PERT, DIST_2D0, VPERT_CORR_2D0, kx=3, ky=3)

    UPERT_CORR_2D = spline_upert(Z_PERT, DIST_2D)
    VPERT_CORR_2D = spline_vpert(Z_PERT, DIST_2D)

    return UVPERT_1D, Z_PERT / 1000, DIST_2D, UPERT_CORR_2D, VPERT_CORR_2D


def get_GW_1D_profile(
    FACT=0.5,
    NRAND=50,
    lambda_max_km=150,
    dlambda_km=1,
    z_min_km=0,
    z_max_km=60,
    dz_km=0.1,
):
    """
    Returns a single 1D vertical profile of u* and v* gravity wave perturbations.

    Parameters:
        FACT         : scaling factor for spectral amplitude
        NRAND        : number of random realisations to draw from
        lambda_max_km: maximum vertical wavelength [km]
        dlambda_km   : wavelength resolution [km]
        z_min_km     : minimum output altitude [km]
        z_max_km     : maximum output altitude [km]
        dz_km        : output altitude resolution [km]

    Returns:
        Z_out_km  : output altitude array [km]
        u_profile : zonal wind perturbation [m/s]
        v_profile : meridional wind perturbation [m/s]
    """
    # --- Physical constants ---
    lambda_b_m = 500
    m_b_m = 2 * np.pi / lambda_b_m
    alpha = 0.62
    N, f = 2 * np.pi / (5 * 60), 2 * np.pi / (18.7 * 60 * 60)
    p, q, s = 2, 3, 2
    SH_m = 7000

    # --- Wavenumber grid ---
    dlambda_m = dlambda_km * 1000
    lambda_min_m = dlambda_m
    lambda_max_m = lambda_max_km * 1000 + lambda_min_m
    dinv_lambda_m = 1 / (lambda_max_m - lambda_min_m)
    inv_lambda_m = np.arange(1 / lambda_max_m, 1 / lambda_min_m + dinv_lambda_m, dinv_lambda_m)
    m_z_m = 2 * np.pi * inv_lambda_m

    # --- Altitude layers ---
    DH_km, DH_max_km = 10, 150
    H_m = np.arange(DH_km, DH_max_km + DH_km, DH_km) * 1000
    m_star_m = 2 * np.pi * np.exp(-(H_m) / (q + s) / SH_m) / 1000

    # --- Spectral amplitudes ---
    Fu_sat = 2 * np.pi * alpha * N**2 / m_z_m**3 / 3
    Fu_turb = 2 * np.pi * alpha * N**2 * (m_b_m / m_z_m) ** (5 / 3) / m_b_m**3 / 3
    Fu_nosat = FACT * 2 * np.pi * alpha * N**2 * (m_z_m / m_star_m[:, None]) ** s / m_star_m[:, None] ** 3 / 3
    Fu_all = np.tile(FACT * Fu_sat, (len(H_m), 1))
    for k in range(len(H_m)):
        Fu_all[k, m_z_m < m_star_m[k]] = Fu_nosat[k, m_z_m < m_star_m[k]]
        Fu_all[k, m_z_m > m_b_m] = FACT * Fu_turb[m_z_m > m_b_m]

    # --- Generate NRAND vertical profiles per layer via IFFT ---
    V_new = np.zeros((len(H_m), NRAND, len(m_z_m) * 2 - 2))
    for k in range(len(H_m)):
        absFU = np.sqrt(Fu_all[k])
        for n in range(NRAND):
            phase = 2 * np.pi * np.random.rand(len(absFU))
            realFU = np.concatenate([absFU * np.cos(phase), absFU[-2:0:-1] * np.cos(phase[-2:0:-1])])
            imagFU = np.concatenate([absFU * np.sin(phase), -absFU[-2:0:-1] * np.sin(phase[-2:0:-1])])
            FUI = realFU + 1j * imagFU
            FUI -= np.mean(FUI)
            fact_ifft = np.sqrt(lambda_max_m - lambda_min_m) / len(FUI)
            V_new[k, n] = np.real(np.fft.ifft(FUI) / fact_ifft)

    # Internal Z_PERT grid (fixed by the IFFT length)
    Z_PERT = np.arange(len(V_new[0, 0])) / (1 / lambda_min_m - 1 / lambda_max_m) / 2

    # --- Tapering window (suppress above ~110 km) ---
    ZT, DZ = 110, 20
    W = np.ones(Z_PERT.shape)
    indz = np.where(Z_PERT / 1000 > ZT)[0]
    W[indz[0] :] = np.exp(-(((Z_PERT[indz[0] :] / 1000 - ZT) / DZ) ** 2))

    # --- Combine layers with Gaussian vertical weighting ---
    a = 1.3015
    sigma = np.sqrt(8 * np.log(a) * 2 * np.pi) / 2.108
    UVPERT_1D = np.zeros((NRAND, len(Z_PERT)))
    for n in range(NRAND):
        for k in range(len(H_m)):
            UVPERT_1D[n] += W * V_new[k, n] * (1 / sigma) * np.exp(-np.log(a) * ((Z_PERT / 1000 - H_m[k] / 1000) / DH_km * 2) ** 2)

    # --- Pick one random realisation for u and v ---
    u_internal = UVPERT_1D[np.random.randint(NRAND)]
    v_internal = UVPERT_1D[np.random.randint(NRAND)]

    # --- Interpolate onto requested output grid ---
    Z_out_km = np.arange(z_min_km, z_max_km + dz_km, dz_km)
    u_profile = np.interp(Z_out_km, Z_PERT / 1000, u_internal)
    v_profile = np.interp(Z_out_km, Z_PERT / 1000, v_internal)

    return Z_out_km, u_profile, v_profile


if __name__ == "__main__":
    # UVPERT_1D2,Z_PERT2,DIST_2D2,UPERT_CORR_2D2,VPERT_CORR_2D2=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D3,Z_PERT3,DIST_2D3,UPERT_CORR_2D3,VPERT_CORR_2D3=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D4,Z_PERT4,DIST_2D4,UPERT_CORR_2D4,VPERT_CORR_2D4=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D5,Z_PERT5,DIST_2D5,UPERT_CORR_2D5,VPERT_CORR_2D5=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D6,Z_PERT6,DIST_2D6,UPERT_CORR_2D6,VPERT_CORR_2D6=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D7,Z_PERT7,DIST_2D7,UPERT_CORR_2D7,VPERT_CORR_2D7=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D8,Z_PERT8,DIST_2D8,UPERT_CORR_2D8,VPERT_CORR_2D8=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D9,Z_PERT9,DIST_2D9,UPERT_CORR_2D9,VPERT_CORR_2D9=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # UVPERT_1D10,Z_PERT10,DIST_2D10,UPERT_CORR_2D10,VPERT_CORR_2D10=get_GW_newparam(DMAX=8000,DK_RANGE=50,FACT=0.5,NRAND=50)
    # breakpoint()

    # The final distance becomes 2*DMAX - DK_RANGE

    UVPERT_1D, Z_PERT, DIST_2D, UPERT_CORR_2D, VPERT_CORR_2D = get_GW_newparam(DMAX=2238, DK_RANGE=50, FACT=0.5, NRAND=50)

    print(Z_PERT.shape)
    print(Z_PERT[0], Z_PERT[-1])

    # uline = UPERT_CORR_2D[:, 1]
    # vline = VPERT_CORR_2D[:, 1]

    # # np.savetxt("data/gws/z_pert.dat", Z_PERT)
    # # np.savetxt("data/gws/u_pert.dat", UPERT_CORR_2D)
    # # np.savetxt("data/gws/v_pert.dat", VPERT_CORR_2D)

    # print(uline.shape)
    # plt.plot(uline, Z_PERT, "o-", label="uline")
    # plt.plot(vline, Z_PERT, "o-", label="vline")
    # plt.plot(UVPERT_1D[0, :], Z_PERT, label="uvpert1d")
    # plt.legend()
    # plt.show()

    # uvpert = UVPERT_1D[:, Z_PERT < 60]
    # zpert = Z_PERT[Z_PERT < 60]
    # for i in range(5):
    #     plt.plot(uvpert[i, :], zpert)
    # plt.show()

    plt.figure(figsize=(9, 7))
    plt.suptitle("Zonal wind speed field perturbations", fontsize=23)

    plt.subplot(2, 1, 1)
    plt.title("U")
    cmp = plt.pcolormesh(DIST_2D, Z_PERT, UPERT_CORR_2D, cmap="bwr", vmin=-30, vmax=30, shading="auto")
    cb = plt.colorbar(cmp)
    cb.set_label(label="u* [m/s]", size=23)
    cb.ax.tick_params(labelsize=18)

    plt.ylabel("Altitude [km]", fontsize=23)
    plt.xlabel("Distance [km]", fontsize=23)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("V")
    cmp = plt.pcolormesh(DIST_2D, Z_PERT, VPERT_CORR_2D, cmap="bwr", vmin=-30, vmax=30, shading="auto")
    cb = plt.colorbar(cmp)
    cb.set_label(label="v* [m/s]", size=23)
    cb.ax.tick_params(labelsize=18)

    plt.ylabel("Altitude [km]", fontsize=23)
    plt.xlabel("Distance [km]", fontsize=23)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()

    plt.tight_layout()
    plt.show(block=True)

    z_km, u, v = get_GW_1D_profile(z_min_km=0, z_max_km=150, dz_km=0.1)
    plt.figure()
    plt.plot(u, z_km)
    plt.plot(v, z_km)
    plt.show()
    # breakpoint()
