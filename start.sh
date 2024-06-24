models=("VGG16" "ResNet18")
# shellcheck disable=SC2068
for model in ${models[@]}; do
    nohup "./run/$model.sh" > "./outs/$model.out" 2>&1 &
done
# hello