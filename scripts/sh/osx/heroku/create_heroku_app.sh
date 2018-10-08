
echo "Script to create and deploy and meteor based heroku application."
echo "$1 was passed as name of application"
if [ ! -z "$2" ]; then
   echo "The path of created app will be inside of $2"
   cd $2
else
    echo "The app will be created in current path '.'"
fi
{

    meteor create $1 &&
    cd $1 &&
    meteor add accounts-ui accounts-facebook accounts-twitter pauli:accounts-linkedin facebook-config-ui twitter-config-ui &&
    git init &&
    git add . &&
    git commit -m "My first commit!" &&
    echo "Meteor app created..." &&
    echo "Registerin on heroku" &&
    heroku login &&
    heroku apps:create $1 &&
    echo "Heroky app creater as $1" &&
    echo "Adding build packs needed for meteor app" &&
    heroku buildpacks:set https://github.com/AdmitHub/meteor-buildpack-horse.git &&
    echo "Buildpack added..." &&
    echo "Creating mongolab database" &&
    heroku addons:create mongolab:sandbox &&
    echo "Database created on heroku" &&
    heroku config | grep MONGODB_URI &&
    heroku config:add MONGO_URL=$MONGODB_URI &&
    heroku config:add ROOT_URL=https://$1.herokuapp.com &&
    echo "Deploing app do heroku" &&
    git push heroku master &&
    echo "Heroku app created...." &&
    echo "Showing logs" &&
    heroku logs --tail
}
