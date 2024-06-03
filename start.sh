models=("VGG13" "VGG19" "ResNet18" "ResNet50")
# shellcheck disable=SC2068
for model in ${models[@]}; do
    nohup "./run/$model.sh" > "./logs/$model.out" 2>&1 &
done