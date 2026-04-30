#!/bin/bash

# Struktur direktori
ROOT_DIR="./html_files"
MEDIA_DIR="${ROOT_DIR}/media"
SOURCE_IMAGES_DIR="./"

# Pastikan direktori utama ada
mkdir -p "$MEDIA_DIR"

# Daftar gambar sumber
IMAGES=("1.jpg" "2.jpg" "3.jpg" "4.jpg" "5.jpg" "GIF-1.gif" "GIF-2.gif" "GIF-3.gif")

# Variasi elemen halaman
TITLES=("Gallery" "Image Collection" "Photo Showcase" "Visual Display" "Picture Gallery" "Media Center" "Image Hub" "Photo Archive")
BACKGROUNDS=("#f0f8ff" "#ffefd5" "#e6e6fa" "#f0fff0" "#fff8dc" "#fdf5e6" "#f5f5dc" "#fffacd")
CSS_STYLES=(
    "text-align: center; font-family: Arial, sans-serif;"
    "text-align: left; font-family: Georgia, serif;"
    "text-align: center; font-family: 'Courier New', monospace;"
    "text-align: justify; font-family: Verdana, sans-serif;"
)
DESCRIPTIONS=("Beautiful image collection" "Stunning visual gallery" "Amazing photo showcase" "Captivating imagery" "Impressive visual display")
CAPTIONS=("Featured image" "Gallery highlight" "Visual showcase" "Photo display" "Image collection")

# Fungsi helper
get_random_element() {
    local arr=("$@")
    local index=$(( RANDOM % ${#arr[@]} ))
    echo "${arr[$index]}"
}

get_random_images() {
    local num_images=$(( RANDOM % 5 + 2 )) # antara 2–6 gambar
    local temp_images=("${IMAGES[@]}")
    local selected_images=()
    for ((i=0; i<num_images && i<${#temp_images[@]}; i++)); do
        local rand_index=$(( RANDOM % ${#temp_images[@]} ))
        selected_images+=("${temp_images[$rand_index]}")
        temp_images=("${temp_images[@]:0:$rand_index}" "${temp_images[@]:$((rand_index+1))}")
    done
    printf '%s\n' "${selected_images[@]}"
}

get_extension() {
    echo "${1##*.}"
}

# Loop untuk generate index1.html sampai index100.html
for i in $(seq 1 100); do
    HTML_FILE="${ROOT_DIR}/index${i}.html"
    PAGE_MEDIA_DIR="${MEDIA_DIR}/index${i}"
    mkdir -p "$PAGE_MEDIA_DIR"

    TITLE=$(get_random_element "${TITLES[@]}")
    BACKGROUND=$(get_random_element "${BACKGROUNDS[@]}")
    STYLE=$(get_random_element "${CSS_STYLES[@]}")
    DESCRIPTION=$(get_random_element "${DESCRIPTIONS[@]}")

    mapfile -t SELECTED_IMAGES < <(get_random_images)
    declare -a RENAMED_IMAGES

    counter=1
    for img in "${SELECTED_IMAGES[@]}"; do
        SRC="${SOURCE_IMAGES_DIR}/${img}"
        EXT=$(get_extension "$img")
        NEW_NAME="media_${i}_${counter}.${EXT}"
        DEST="${PAGE_MEDIA_DIR}/${NEW_NAME}"

        if [ -f "$SRC" ]; then
            cp "$SRC" "$DEST"
        else
            echo "Warning: ${SRC} tidak ditemukan."
        fi
        RENAMED_IMAGES+=("$NEW_NAME")
        ((counter++))
    done

    # Generate HTML
    cat > "$HTML_FILE" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${TITLE} ${i}</title>
<style>
body {
    background-color: ${BACKGROUND};
    ${STYLE}
    margin: 20px;
    padding: 20px;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-top: 20px;
}
.image-item {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.image-item:hover {
    transform: scale(1.05);
}
.image-item img {
    width: 100%;
    height: auto;
    border-radius: 4px;
}
.image-caption {
    margin-top: 8px;
    font-size: 14px;
    color: #666;
}
h1 {
    color: #333;
    margin-bottom: 10px;
}
.description {
    color: #666;
    font-style: italic;
    margin-bottom: 20px;
}
</style>
</head>
<body>
<div class="container">
<h1>${TITLE} ${i}</h1>
<p class="description">${DESCRIPTION}</p>
<div class="image-grid">
EOF

    for renamed_img in "${RENAMED_IMAGES[@]}"; do
        CAPTION=$(get_random_element "${CAPTIONS[@]}")
        cat >> "$HTML_FILE" << EOF
<div class="image-item">
    <img src="media/index${i}/${renamed_img}" alt="${renamed_img}">
    <div class="image-caption">${CAPTION} - ${renamed_img}</div>
</div>
EOF
    done

    cat >> "$HTML_FILE" << EOF
</div>
</div>
</body>
</html>
EOF

    echo "✔ index${i}.html dibuat dengan ${#RENAMED_IMAGES[@]} gambar di ${PAGE_MEDIA_DIR}"
    unset RENAMED_IMAGES
done

echo ""
echo "========================================="
echo "Semua file HTML dan media berhasil dibuat!"
echo "Root: ${ROOT_DIR}"
echo "Media: ${MEDIA_DIR}/indexN/"
echo "========================================="
