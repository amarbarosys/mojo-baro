#!/usr/bin/env bash
# Install this app's icon and launcher entry for the current user (KDE menu, Wayland icon).
set -euo pipefail
d="$(cd "$(dirname "$0")/.." && pwd)"; app=mojo-baro; t=~/.local/share/icons/hicolor
mkdir -p $t/scalable/apps ~/.local/share/applications
[ -f $t/index.theme ] || install -m644 /usr/share/icons/hicolor/index.theme $t/index.theme
install -m644 "$d/assets/icon/$app.svg" $t/scalable/apps/$app.svg
install -m644 "$d/assets/icon/$app.desktop" ~/.local/share/applications/$app.desktop
gtk-update-icon-cache -f $t
kbuildsycoca6
echo "installed $app icon and desktop entry"
