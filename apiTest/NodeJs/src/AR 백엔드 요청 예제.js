const fs = require("fs");

async function requestARbackend(formData) {
    const req = await fetch("http://localhost/bodymea/", {
        method: "post",
        body: formData,
    });

    return await req.json();
}
const formData = new FormData();
const file = fs.readFileSync("res/test.jpg");
formData.append("anaFile", new Blob([file]));
formData.append("personKey", 174);

requestARbackend(formData);
