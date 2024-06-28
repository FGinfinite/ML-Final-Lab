models=("VGG16" "ResNet18" "VGG16-aug" "ResNet18-aug")
# shellcheck disable=SC2068

for model in ${models[@]}; do
    touch "./outs/$model.out"
done
for model in ${models[@]}; do
    nohup "./run/$model.sh" > "./outs/$model.out" 2>&1 &
done
# hello