function signuppage() {
    document.getElementById("signup").style.display = "block";
    document.getElementById("login").style.display = "none";
}

function signinpage() {
    document.getElementById("signup").style.display = "none";
    document.getElementById("login").style.display = "block";
}

function signup() {
    let pass = document.getElementById("Spassword").value;
    let Cpass = document.getElementById("SCpassword").value;
    let email = document.getElementById("Semail").value;
    let user = document.getElementById("Susername").value;
    let terms = document.getElementById("terms").checked;

    if(email && user && pass && Cpass){
        if(terms){
            if(pass == Cpass){
                localStorage.setItem(user,pass);
                window.location.href = "login.html"
            }
            else{
                alert("Password in not matched")
            }
        }
        else{
            alert("check the terms & conditions")
        }
    }

}

function signin() {
    let pass = document.getElementById("Lpassword").value;
    let user = document.getElementById("Lusername").value;

    let Cpass = localStorage.getItem(user);

    if(pass && user){
        if(Cpass){
            if(pass == Cpass){
                localStorage.setItem("logged",true)
                localStorage.setItem("user",user)
                window.location.href = "Index.html"
            }
            else{
                alert("incorrect password")
            }
        }
        else{
            alert("no user found")
        }
    }
}