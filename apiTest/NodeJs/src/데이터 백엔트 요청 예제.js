async function requestPost(url, data) {
    const req = await fetch(url, {
        method: "post",
        headers: {
            "Content-Type": "application/json",
        },
        body: data,
    });

    return await req.json();
}

async function requestGet(url) {
    const req = await fetch(url, { method: "get" });
    return await req.json();
}
