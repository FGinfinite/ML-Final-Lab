models=("VGG13" "VGG16" "ResNet18")
for model in ${models[@]}; do
    touch "./outs/$model.out"
done