models=("VGG11" "VGG13" "VGG16" "VGG19" "ResNet18" "ResNet34" "ResNet50" "ResNet101" "ResNet152")
for model in ${models[@]}; do
    touch "./logs/$model.out"
done