import torch


def get_gen_loss(crit_fake_pred):
    gen_loss = -torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss


def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    disc_loss_real = adv_criterion(disc_X(real_X), torch.ones_like(disc_X(real_X)))
    disc_loss_fake = adv_criterion(
        disc_X(fake_X.detach()), torch.zeros_like(disc_X(fake_X.detach()))
    )
    disc_loss = (disc_loss_real + disc_loss_fake) / 2

    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    fake_Y = gen_XY(real_X)
    disc_pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_pred, torch.ones_like(disc_Y(real_X)))

    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)

    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)

    return cycle_loss, cycle_X


def get_gen_loss(
    real_A,
    real_B,
    gen_AB,
    gen_BA,
    disc_A,
    disc_B,
    adv_criterion,
    identity_criterion,
    cycle_criterion,
    lambda_identity=0.1,
    lambda_cycle=10,
):
    adv_loss_xy, fake_B = get_gen_adversarial_loss(
        real_A, disc_B, gen_AB, adv_criterion
    )
    adv_loss_yx, fake_A = get_gen_adversarial_loss(
        real_B, disc_A, gen_BA, adv_criterion
    )

    identity_loss_xy, _ = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_xy = lambda_identity * identity_loss_xy
    identity_loss_yx, _ = get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss_yx = lambda_identity * identity_loss_yx

    cycle_loss_xy, _ = get_cycle_consistency_loss(
        real_A, fake_B, gen_BA, cycle_criterion
    )
    cycle_loss_xy = lambda_cycle * cycle_loss_xy
    cycle_loss_yx, _ = get_cycle_consistency_loss(
        real_B, fake_A, gen_AB, cycle_criterion
    )
    cycle_loss_yx = lambda_cycle * cycle_loss_yx

    gen_loss = (
        adv_loss_xy
        + adv_loss_yx
        + identity_loss_xy
        + identity_loss_yx
        + cycle_loss_xy
        + cycle_loss_yx
    )

    return gen_loss, fake_A, fake_B
