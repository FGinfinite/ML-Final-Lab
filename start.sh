models=("VGG16" "VGG19" "ResNet18" "ResNet34")
# shellcheck disable=SC2068
for model in ${models[@]}; do
    nohup "./run/$model.sh" > "./logs/$model.out" 2>&1 &
done