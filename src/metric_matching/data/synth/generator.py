import torch


def generate_circle_sine_wave_aligned(
    num_points=100,
    frequency=1,
    radius=1.0,
    sampler="regular",
    sorted=True,
    device="cpu",
):
    """
    Generate 2D data for a sine wave wrapped around a circle with proper alignment.

    Returns x,y torch tensors shape (num_points,) on `device`.
    """
    if sampler == "regular":
        theta = (
            torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
            if num_points > 1
            else torch.tensor([0.0], device=device)
        )
        if not sorted:
            theta = theta[torch.randperm(len(theta), device=device)]
    elif sampler == "random":
        theta = torch.rand(num_points, device=device) * 2 * torch.pi
        if sorted:
            theta = torch.sort(theta)[0]
    else:
        raise ValueError("Invalid sampler")

    if frequency == 0:
        r = torch.full_like(theta, radius, device=device)
    else:
        r = radius + 0.3 * torch.sin(frequency * theta)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    # return points as a single (N,2) tensor: columns [x, y]
    pts = torch.stack([x, y], dim=-1)
    return pts


def generate_sphere_uniform(
    num_points: int,
    radius: float = 1.0,
    d: int = 2,
    D: int = 3,
    noise_level: float = 0.0,
    device=None,
):
    """
    Generate points uniformly on the surface of an intrinsically `d`-dimensional sphere
    embedded in `D` dimensions using the normalisation method.

    Args:
        num_points: number of points to generate
        radius: radius of the sphere
        d: intrinsic dimension (default: 2)
        D: extrinsic dimension (default: 3)
        device: torch device
    Returns:
        points: (num_points, D) tensor of points on the sphere
    """
    if device is None:
        device = torch.device("cpu")

    if D < d + 1:
        raise ValueError(
            "Extrinsic dimension D must be at least d + 1 to embed the sphere."
        )

    base = torch.randn((num_points, d + 1), device=device)
    base = base / base.norm(dim=1, keepdim=True) * radius

    points = torch.zeros((num_points, D), device=device)
    points[:, : d + 1] = base
    if noise_level > 0.0:
        noise = torch.randn((num_points, D), device=device) * noise_level
        points += noise
    return points
