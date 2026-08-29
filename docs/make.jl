import Bornsampling
using Documenter

DocMeta.setdocmeta!(
    Bornsampling,
    :DocTestSetup,
    quote
        import Bornsampling
        const FiniteMPS = Bornsampling.FiniteMPS
        const TK = Bornsampling.TK
        const Random = Bornsampling.Random
    end;
    recursive=true,
)

makedocs(
    modules=[Bornsampling],
    authors="Qiaoyi Li",
    sitename="Bornsampling.jl",
    remotes=nothing,
    checkdocs=:exports,
    format=Documenter.HTML(
        canonical="https://Qiaoyi-Li.github.io/Bornsampling.jl",
        edit_link="main",
        repolink="https://github.com/Qiaoyi-Li/Bornsampling.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Implementation" => "implementation.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo="github.com/Qiaoyi-Li/Bornsampling.jl.git",
    devbranch="main",
)
