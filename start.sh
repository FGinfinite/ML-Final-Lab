models=("VGG13" "VGG16" "ResNet18" "VGG13-aug" "VGG16-aug" "ResNet18-aug")
# shellcheck disable=SC2068
for model in ${models[@]}; do
    nohup "./run/$model.sh" > "./outs/$model.out" 2>&1 &
done
# hello