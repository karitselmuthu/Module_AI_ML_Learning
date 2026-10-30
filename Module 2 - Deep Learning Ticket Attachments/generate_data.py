"""Create a reproducible, synthetic attachment dataset for the Module 2 labs."""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "generated"
LABELS = ("account", "billing", "technical")
SEED = 42

SCENARIOS = {
    "account": [
        ("Sign in blocked", "Password rejected after reset"),
        ("Verify your email", "Confirmation link has expired"),
        ("Two factor code", "The login code never arrived"),
        ("Account locked", "Too many sign in attempts"),
        ("Profile settings", "Change the primary email"),
        ("Workspace access", "Invite a new team member"),
        ("Reset password", "Recovery link is invalid"),
        ("Security settings", "Replace the old phone number"),
        ("User profile", "Update the display name"),
        ("Team permissions", "Remove a former employee"),
        ("Sign on", "Authentication redirects in a loop"),
        ("New device", "Approve this device to sign in"),
        ("Account recovery", "I lost my authenticator phone"),
        ("Email verification", "The code is no longer valid"),
        ("Organization owner", "Transfer workspace ownership"),
        ("Member invitation", "Invitation link is not working"),
        ("Login session", "Sign out all active devices"),
        ("Access request", "Restore my disabled profile"),
        ("Username change", "Save my new account name"),
        ("Admin console", "Grant a teammate access"),
    ],
    "billing": [
        ("Invoice total", "The charged amount looks wrong"),
        ("Payment failed", "The bank says the charge cleared"),
        ("Tax details", "Update the VAT number on invoice"),
        ("Subscription", "Cancel before the next renewal"),
        ("Refund request", "Return an accidental payment"),
        ("Receipt", "Download last month's receipt"),
        ("Credit card", "Replace the expired payment card"),
        ("Pricing", "Explain the extra seat charge"),
        ("Billing address", "Correct the address on our bill"),
        ("Annual plan", "Switch the subscription to yearly"),
        ("Discount", "Coupon did not apply at checkout"),
        ("Statement", "Two charges appear for one order"),
        ("Invoice correction", "Change our company name on bill"),
        ("Payment history", "Show all earlier transactions"),
        ("Renewal notice", "The billed date is too early"),
        ("Tax exemption", "Remove sales tax from invoice"),
        ("Plan downgrade", "Lower next month's charge"),
        ("Wire transfer", "Payment is missing from ledger"),
        ("Credit note", "Issue credit for unused seats"),
        ("Quote request", "Price ten new licenses"),
    ],
    "technical": [
        ("Server error", "Request returned status 500"),
        ("Upload failed", "File stops before completion"),
        ("App crash", "Screen closes when report opens"),
        ("Sync failure", "Integration has stopped syncing"),
        ("Export problem", "CSV download is empty"),
        ("Timeout", "Save request takes too long"),
        ("Search broken", "No results for existing records"),
        ("Webhook", "The event was sent twice"),
        ("Chart bug", "Graph displays the wrong total"),
        ("Browser issue", "Page freezes in Firefox"),
        ("Data import", "Import stops at row three"),
        ("API response", "Endpoint returns invalid JSON"),
        ("Image preview", "Uploaded file looks corrupted"),
        ("Notification bug", "Alerts arrive several hours late"),
        ("Calendar view", "Changes do not appear after save"),
        ("Pagination", "API skips some result pages"),
        ("Mobile issue", "Button does nothing on phone"),
        ("Scheduled job", "The task did not run overnight"),
        ("PDF export", "The right side is cut off"),
        ("Dashboard", "The page stays blank on load"),
    ],
}


def split_for_scenario(index: int) -> str:
    if index < 12:
        return "train"
    if index < 16:
        return "validation"
    return "test"


def draw_attachment(label: str, title: str, detail: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    width, height = 224, 160
    background = rng.choice(("#eef1f5", "#f2f3f5", "#ebeff2"))
    accent = rng.choice(("#456b88", "#526b8f", "#5b7488"))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=10)
    xshift, yshift = rng.randint(-3, 3), rng.randint(-2, 2)
    draw.rounded_rectangle((8 + xshift, 8 + yshift, 216 + xshift, 152 + yshift),
                           radius=7, fill="#ffffff", outline="#c9d1d8", width=1)
    draw.rounded_rectangle((8 + xshift, 8 + yshift, 216 + xshift, 34 + yshift),
                           radius=6, fill=accent)
    draw.text((18 + xshift, 14 + yshift), "SUPPORT ATTACHMENT", fill="white", font=small)
    draw.text((18 + xshift, 43 + yshift), title, fill="#18232d", font=font)
    for line_number, line in enumerate(textwrap.wrap(detail, 25)[:2]):
        draw.text((18 + xshift, 67 + yshift + 16 * line_number), line,
                  fill="#32404b", font=small)

    # UI elements differ by the pictured task; colors and position vary independently.
    if label == "account":
        draw.ellipse((169, 60, 191, 82), outline=accent, width=3)
        draw.arc((157, 77, 203, 117), start=185, end=355, fill=accent, width=4)
        draw.rounded_rectangle((18, 108, 121, 123), radius=2, outline="#b5bec7")
        draw.rounded_rectangle((18, 128, 121, 143), radius=2, outline="#b5bec7")
        draw.text((138, 111), "USER", fill=accent, font=small)
    elif label == "billing":
        draw.rectangle((163, 58, 202, 96), outline=accent, width=3)
        draw.line((170, 70, 195, 70), fill=accent, width=3)
        draw.line((170, 82, 190, 82), fill=accent, width=3)
        draw.line((18, 110, 204, 110), fill="#aeb8c1", width=1)
        draw.text((19, 116), "AMOUNT", fill="#52606c", font=small)
        draw.text((143, 116), f"{rng.randint(20, 900)}.00", fill=accent, font=small)
        draw.line((18, 139, 204, 139), fill="#aeb8c1", width=1)
    else:
        draw.polygon(((183, 55), (158, 98), (208, 98)), outline=accent, width=4)
        draw.line((183, 69, 183, 83), fill=accent, width=4)
        draw.ellipse((181, 88, 185, 92), fill=accent)
        draw.rounded_rectangle((18, 106, 204, 142), radius=3, fill="#e9edf1")
        draw.text((27, 112), "ERROR LOG", fill=accent, font=small)
        draw.text((27, 126), f"code {rng.choice((408, 500, 503))}", fill="#52606c", font=small)
    return image


def generate(output_dir: Path = DEFAULT_OUTPUT, variants: int = 4) -> Path:
    if variants < 1:
        raise ValueError("variants must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_index, (label, scenarios) in enumerate(SCENARIOS.items()):
        for scenario_index, (title, detail) in enumerate(scenarios):
            split = split_for_scenario(scenario_index)
            scenario_id = f"{label}-{scenario_index:02d}"
            for variant in range(variants):
                filename = f"{scenario_id}-{variant:02d}.png"
                relative_path = Path("images") / label / filename
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                draw_attachment(label, title, detail,
                                SEED + class_index * 10000 + scenario_index * 100 + variant).save(destination)
                rows.append((relative_path.as_posix(), label, split, scenario_id))
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "label", "split", "scenario_id"))
        writer.writerows(rows)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=4)
    args = parser.parse_args()
    print(generate(args.output, args.variants))
