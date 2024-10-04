# Will loop over all images passed in via glob and move them into either True or False directories
#
# Usage:
#    main.sh <filepath glob>
#
# Examples
#		./apps/select_img_true_false.py/main.sh "storage/img/training/cars/*.jpg"


globber="$1"
directory=$(dirname "$1")
echo "$globber", "$directory"
if [[ "$globber" != *"*"* ]]; then
	echo "Please include a glob in the filepath glob... I can't see a * in there!"
	exit 1
fi
if [[ ! -d $directory ]]; then
	echo "Please give a valid directory: '$directory' cannot be found"
	exit 1
fi

mkdir -p True
mkdir -p False

for fp in $globber;
do
	open "$fp" &
	sleep 1
	osascript -e 'tell application "Preview" to quit' &
	echo "Is this a good image [$fp]?"
	read -p "[y/n]" ans
	if [[ "$ans" == "r" ]]; then
		open "$fp"
	fi
	while [[ "$ans" != "y" ]] && [[ "$ans" != "n" ]]; do
		read -p "please enter [y/n] (you entered '$ans') : " ans
	done

	if [[ "$ans" == "y" ]]; then
		mv "$fp" "$directory/True/" &
	else
		mv "$fp" "$directory/False/" &
	fi

done
